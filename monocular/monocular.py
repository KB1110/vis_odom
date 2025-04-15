import cv2 as cv
import numpy as np
from .ssc import ssc


class MonocularVisualOdometry:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        image_height: int,
        image_width: int,
        starting_pose: np.ndarray = np.zeros((3, 1)),
    ) -> None:
        self.__INTRINSICS = camera_matrix
        self.__DISTORTION_COEFF = distortion_coeffs
        self.__OPTIMAL_INTRINSICS, self.__ROI = cv.getOptimalNewCameraMatrix(
            self.__INTRINSICS,
            self.__DISTORTION_COEFF,
            (image_width, image_height),
            1,
            (image_width, image_height),
        )
        self.__PROJECTION_MATRIX = np.concatenate(
            (self.__OPTIMAL_INTRINSICS, starting_pose), axis=1
        )
        self.__homogeneous_camera_pose = np.eye(4)
        self.__homogeneous_camera_pose[:3, 3] = starting_pose.flatten()

        self.__orb = cv.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=4,
            scoreType=cv.ORB_HARRIS_SCORE,
            patchSize=31,
        )

        self.__bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        self.__flann = cv.FlannBasedMatcher(index_params, search_params)

        self.__roi_x, self.__roi_y, self.__roi_w, self.__roi_h = self.__ROI

        # Kalman Filter Parameters
        # Initial state
        self.__X = np.zeros((6, 1))  # [x, y, z, vx, vy, vz]
        self.__X[:3, :] = starting_pose

        # Observation matrix (3x6): we only observe positions
        self.__H = np.zeros((3, 6))
        self.__H[0, 0] = 1
        self.__H[1, 1] = 1
        self.__H[2, 2] = 1

        # Covariances
        self.__P = np.eye(6) * 0.8  # State covariance
        self.__P[:3, 3:] = np.eye(3) * 0.4
        self.__P[3:, :3] = np.eye(3) * 0.4
        self.__P[3:, 3:] = np.eye(3) * 2.5

        self.__R = np.eye(3) * 0.5  # Measurement noise covariance

    def compute_keypoints_and_descriptors(self, gray: np.ndarray):

        self.__gray = cv.undistort(
            gray,
            self.__INTRINSICS,
            self.__DISTORTION_COEFF,
            None,
            self.__OPTIMAL_INTRINSICS,
        )
        self.__gray = self.__gray[
            self.__roi_y : self.__roi_y + self.__roi_h,
            self.__roi_x : self.__roi_x + self.__roi_w,
        ]
        self.__gray = cv.flip(self.__gray, 1)

        keypoints = self.__orb.detect(self.__gray, None)

        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:500]

        keypoints = ssc(keypoints, 250, 0.5, self.__gray.shape[1], self.__gray.shape[0])

        keypoints, descriptors = self.__orb.compute(self.__gray, keypoints)

        return keypoints, descriptors

    def compute_matches(
        self,
        prev_descriptor: np.ndarray,
        curr_descriptor: np.ndarray,
        prev_keypoints: tuple,
        curr_keypoints: tuple,
        lowe_ratio: float = 0.8,
        matcher: int = 0,
    ):
        if matcher == 0:
            matches = self.__bf.knnMatch(prev_descriptor, curr_descriptor, k=2)
        elif matcher == 1:
            matches = self.__flann.knnMatch(
                np.float32(prev_descriptor), np.float32(curr_descriptor), k=2
            )
        good_matches = []

        # Lowe's Ratio Test
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

        prev_good_keypoints = np.array(
            [prev_keypoints[m.queryIdx].pt for m in good_matches]
        )
        curr_good_keypoints = np.array(
            [curr_keypoints[m.trainIdx].pt for m in good_matches]
        )

        return good_matches, prev_good_keypoints, curr_good_keypoints

    def compute_pose(
        self,
        prev_good_keypoints: np.ndarray,
        curr_good_keypoints: np.ndarray,
        time_step: float = 0.01,
        post_process: int = 0,
    ):

        essential_matrix, mask = cv.findEssentialMat(
            prev_good_keypoints,
            curr_good_keypoints,
            self.__OPTIMAL_INTRINSICS,
            cv.RANSAC,
            0.999,
            1.0,
        )
        transformation_matrix = self.__compute_transformation_matrix(
            essential_matrix, prev_good_keypoints, curr_good_keypoints
        )

        self.__homogeneous_camera_pose = np.matmul(
            transformation_matrix, self.__homogeneous_camera_pose
        )
        cam_pose = self.__homogeneous_camera_pose[:3, 3]

        if post_process == 1:
            self.__kalman_filter_update(cam_pose, time_step)

        return self.__homogeneous_camera_pose

    def __compute_transformation_matrix(
        self,
        essential_matrix: np.ndarray,
        prev_good_keypoints: np.ndarray,
        curr_good_keypoints: np.ndarray,
    ):
        R1, R2, t = cv.decomposeEssentialMat(essential_matrix)

        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = np.ndarray.flatten(t)

        T2 = np.eye(4)
        T2[:3, :3] = R2
        T2[:3, 3] = np.ndarray.flatten(t)

        T3 = np.eye(4)
        T3[:3, :3] = R1
        T3[:3, 3] = np.ndarray.flatten(-t)

        T4 = np.eye(4)
        T4[:3, :3] = R2
        T4[:3, 3] = np.ndarray.flatten(-t)

        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.__OPTIMAL_INTRINSICS, np.zeros((3, 1))), axis=1)

        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []

        for projection, transformation in zip(projections, transformations):
            homogeneous_Q1 = cv.triangulatePoints(
                self.__PROJECTION_MATRIX,
                projection,
                prev_good_keypoints.T,
                curr_good_keypoints.T,
            )
            homogeneous_Q2 = transformation @ homogeneous_Q1

            # Homogeneous to Euclidean
            euclidean_Q1 = homogeneous_Q1[:3, :] / homogeneous_Q1[3, :]
            euclidean_Q2 = homogeneous_Q2[:3, :] / homogeneous_Q2[3, :]

            total_sum = sum(euclidean_Q2[2, :] > 0) + sum(euclidean_Q1[2, :] > 0)
            relative_scale = np.mean(
                np.linalg.norm(euclidean_Q1.T[:-1] - euclidean_Q1.T[1:], axis=-1)
                / np.linalg.norm(euclidean_Q2.T[:-1] - euclidean_Q2.T[1:], axis=-1)
            )

            positives.append(total_sum + relative_scale)

        max_positive = np.argmax(positives)

        if max_positive == 0:
            rotation_matrix = R1
            translation_vector = np.ndarray.flatten(t)
        elif max_positive == 1:
            rotation_matrix = R2
            translation_vector = np.ndarray.flatten(t)
        elif max_positive == 2:
            rotation_matrix = R1
            translation_vector = np.ndarray.flatten(-t)
        elif max_positive == 3:
            rotation_matrix = R2
            translation_vector = np.ndarray.flatten(-t)

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        transformation_matrix[:3, 3] = translation_vector

        return transformation_matrix

    def __kalman_filter_update(self, cam_pose, time_step):
        # State transition matrix (6x6)
        ACCEL = 0.01
        SIGMA_ACCEL = 0.75

        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * time_step

        # Process Noise
        W = np.zeros((6, 1))
        W[0] = ((time_step**2) / 2) * np.random.randn()
        W[1] = ((time_step**2) / 2) * np.random.randn()
        W[2] = ((time_step**2) / 2) * np.random.randn()

        W = (ACCEL**2) * W

        Q = np.eye(6) * ((time_step**4) / 4)
        Q[:3, 3:] = np.eye(3) * ((time_step**3) / 2)
        Q[3:, :3] = np.eye(3) * ((time_step**3) / 2)
        Q[3:, 3:] = np.eye(3) * (time_step**2)

        Q = (SIGMA_ACCEL**2) * Q

        # Predict
        self.__X = (F @ self.__X) + W
        self.__P = (F @ self.__P @ F.T) + Q

        # Measurement
        Z = cam_pose.reshape(3, 1)

        # Innovation
        Y = Z - (self.__H @ self.__X)

        # Kalman Gain
        S = self.__H @ self.__P @ self.__H.T + self.__R
        K = (self.__P @ self.__H.T) @ np.linalg.inv(S)

        # Update
        self.__X = self.__X + (K @ Y)
        self.__P = (np.eye(6) - K @ self.__H) @ self.__P

        # Smoothed position
        smoothed_cam_pose = self.__X[:3].flatten()
        self.__homogeneous_camera_pose[:3, 3] = smoothed_cam_pose
