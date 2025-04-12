import cv2 as cv
import numpy as np
import ssc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

INTRINSICS = np.array(
    [
        [166.13029993, 0.0, 319.49999036],
        [0.0, 282.319318, 239.50000072],
        [0.0, 0.0, 1.0],
    ]
)

OPTIMAL_INTRINSICS = np.array(
    [
        [144.35171509, 0.0, 312.17998081],
        [0.0, 241.51057434, 239.00072201],
        [0.0, 0.0, 1.0],
    ]
)

DISTORTION_COEFF = np.array(
    [
        [
            -2.27010164e-02,
            1.10733614e-04,
            1.33925610e-05,
            -1.87437026e-03,
            -1.09357096e-07,
        ]
    ]
)

ROI = np.array((8, 31, 620, 418))  # x, y, w, h

LOWE_RATIO = 0.8

homogeneous_camera_pose = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

PROJECTION_MATRIX = np.concatenate(
    (INTRINSICS, np.zeros((3, 1))), axis = 1 #Starting pose considered to be at origin
)

orb = cv.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=4,
    scoreType=cv.ORB_HARRIS_SCORE,
    patchSize=31,
)

# Brute Force Matcher
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False)

x, y, w, h = ROI

trajectory = []

cam_pose = np.zeros((3, 1))
cam_transformation = np.eye(4)

# Kalman Filter Parameters
dt = 0.01  # time step (can be adapted from timestamps)

# State transition matrix (6x6)
A = np.eye(6)
A[0, 3] = dt
A[1, 4] = dt
A[2, 5] = dt

# Observation matrix (3x6): we only observe positions
H = np.zeros((3, 6))
H[0, 0] = 1
H[1, 1] = 1
H[2, 2] = 1

# Initial state
X = np.zeros((6, 1))  # [x, y, z, vx, vy, vz]

# Covariances
P = np.eye(6) * 0.8         # State covariance
Q = np.eye(6) * 0.01        # Process noise covariance
R = np.eye(3) * 0.8         # Measurement noise covariance

frame = cv.VideoCapture(0, cv.CAP_V4L2)

first_frame = True

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
(trajectory_plot,) = ax.plot([], [], [], label="Camera Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim3d(-10, 10)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(0, 20)

ax.legend()

plt.draw()
plt.pause(0.001)

while True:
    ret, img = frame.read()
    if not ret:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.undistort(gray, INTRINSICS, DISTORTION_COEFF, None, OPTIMAL_INTRINSICS)
    gray = gray[y : y + h, x : x + w]
    gray = cv.flip(gray, 1)
    
    keypoints_curr = orb.detect(gray, None)
    
    keypoints_curr = sorted(
        keypoints_curr, key=lambda x: x.response, reverse=True
    )[:500]
        
    selected_keypoints_curr = ssc.ssc(
        keypoints_curr, 250, 0.5, gray.shape[1], gray.shape[0]
    )
        
    selected_keypoints_curr, descriptor_curr = orb.compute(gray, selected_keypoints_curr)

    if not first_frame:
        matches = bf.knnMatch(descriptor_prev, descriptor_curr, k = 2)
        good_matches = []
        
        #Lowe's Ratio Test
        for m, n in matches:
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
        
        good_prev_kp = np.array(
            [selected_keypoints_prev[m.queryIdx].pt for m in good_matches]
        )      
        good_curr_kp = np.array(
            [selected_keypoints_curr[m.trainIdx].pt for m in good_matches]
        )
        
        draw_params = dict(
            matchColor = -1,
            singlePointColor = None,
            matchesMask = None,
            flags = 2
        )
        
        matched_img = cv.drawMatches(
            prev_gray, selected_keypoints_prev, gray, selected_keypoints_curr, good_matches, None, **draw_params 
        )
        
        cv.imshow("Matches", matched_img)
        
        essential_matrix, mask = cv.findEssentialMat(
            good_prev_kp, good_curr_kp, INTRINSICS, cv.RANSAC, 0.999, 1.0
        )
        
        R1, R2, t = cv.decomposeEssentialMat(essential_matrix)
        
        H1 = np.eye(4)
        H1[:3, :3] = R1
        H1[:3, 3] = np.ndarray.flatten(t)
        
        H2 = np.eye(4)
        H2[:3, :3] = R2
        H2[:3, 3] = np.ndarray.flatten(t)
        
        H3 = np.eye(4)
        H3[:3, :3] = R1
        H3[:3, 3] = np.ndarray.flatten(-t)
        
        H4 = np.eye(4)
        H4[:3, :3] = R2
        H4[:3, 3] = np.ndarray.flatten(-t)
        
        transformations = [H1, H2, H3, H4]
        
        K = np.concatenate(
            (INTRINSICS, np.zeros((3, 1))), axis = 1
        )
        
        projections = [K @ H1, K @ H2, K @ H3, K @ H4]
        
        positives = []
        
        for projection, transformation in zip(projections, transformations):
            homogeneous_Q1 = cv.triangulatePoints(PROJECTION_MATRIX, projection, good_prev_kp.T, good_curr_kp.T)
            homogeneous_Q2 = transformation @ homogeneous_Q1
            
            # Homogeneous to Euclidean
            euclidean_Q1 = homogeneous_Q1[:3, :] / homogeneous_Q1[3, :]
            euclidean_Q2 = homogeneous_Q2[:3, :] / homogeneous_Q2[3, :]
            
            total_sum = sum(euclidean_Q2[2, :] > 0) + sum(euclidean_Q1[2, :] > 0)
            relative_scale = np.mean(
                np.linalg.norm(euclidean_Q1.T[:-1] - euclidean_Q1.T[1:], axis = -1) /
                np.linalg.norm(euclidean_Q2.T[:-1] - euclidean_Q2.T[1:], axis = -1)
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
        
        print(transformation_matrix)
        
        homogeneous_camera_pose = np.matmul(transformation_matrix, homogeneous_camera_pose)
        cam_pose = homogeneous_camera_pose[:3, 3]
        
        # Predict
        X = A @ X
        P = A @ P @ A.T + Q

        # Measurement
        Z = cam_pose.reshape(3, 1)

        # Kalman Gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Update
        Y = Z - H @ X  # Innovation
        X = X + K @ Y
        P = (np.eye(6) - K @ H) @ P

        # Smoothed position
        smoothed_cam_pose = X[:3].flatten()
        homogeneous_camera_pose[:3, 3] = smoothed_cam_pose

        trajectory.append(smoothed_cam_pose)
        
    prev_gray = gray
    selected_keypoints_prev = selected_keypoints_curr
    descriptor_prev = descriptor_curr
    
    cv.imshow("Cam_feed", gray)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    
    first_frame = False
    
    # Update the trajectory plot
    trajectory_np = np.array(trajectory)  # Convert trajectory to a NumPy array
    if trajectory_np.shape[0] > 1:  # Ensure there are at least two points to plot
        trajectory_plot.set_data(trajectory_np[:, 0], trajectory_np[:, 1])  # X and Y
        trajectory_plot.set_3d_properties(trajectory_np[:, 2])  # Z
        plt.draw()
        plt.pause(0.001)

frame.release()
cv.destroyAllWindows()
