from monocular.monocular import MonocularVisualOdometry
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


def load_kitti_sequence(dataset_path: str, sequence: str):
    KITTI_GRAYSCALE_DATASET_PATH = "data_odometry_gray/dataset/sequences/"
    KITTI_POSE_PATH = "data_odometry_poses/dataset/poses/"
    KITTI_SEQUENCE = sequence

    images = []
    poses = []
    timestamps = []

    images_folder = os.path.join(
        dataset_path, KITTI_GRAYSCALE_DATASET_PATH, KITTI_SEQUENCE, "image_0/"
    )
    pose_file = os.path.join(dataset_path, KITTI_POSE_PATH, f"{KITTI_SEQUENCE}.txt")
    calib_file = os.path.join(
        dataset_path, KITTI_GRAYSCALE_DATASET_PATH, KITTI_SEQUENCE, "calib.txt"
    )
    timestamps_file = os.path.join(
        dataset_path, KITTI_GRAYSCALE_DATASET_PATH, KITTI_SEQUENCE, "times.txt"
    )

    for filename in sorted(os.listdir(images_folder)):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            images.append(image_path)

    with open(pose_file, "r") as f:
        for line in f.readlines():
            pose = [float(x) for x in line.split()]
            pose = np.array(pose).reshape(3, 4)
            poses.append(pose)

    with open(calib_file, "r") as f:
        for line in f.readlines():
            if "P0" in line:
                calibration_matrix = [float(x) for x in line.split()[1:]]
                break

    with open(timestamps_file, "r") as f:
        for line in f.readlines():
            timestamp = [float(x) for x in line.split()[:]]
            timestamps.append(timestamp[0])

    calibration_matrix = np.array(calibration_matrix).reshape(3, 4)
    intrinsics, rotation_matrix, translation_vector, _, _, _, _ = (
        cv.decomposeProjectionMatrix(calibration_matrix)
    )
    starting_pose = translation_vector[:3, :]

    distortion_coeffs = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )

    return images, poses, timestamps, intrinsics, starting_pose, distortion_coeffs


def main():
    KITTI_DATASET_PATH = "/home/kb/kitti-dataset/"
    KITTI_SEQUENCE = "00"

    FINAL_TRANSFORM = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ) @ np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    LOWE_RATIO = 0.8

    images, poses, timestamps, intrinsics, starting_pose, distortion_coeffs = (
        load_kitti_sequence(KITTI_DATASET_PATH, KITTI_SEQUENCE)
    )
    img = cv.imread(images[0])

    trajectory = []
    ground_truth = []
    ground_truth_pose = np.concatenate((starting_pose, np.array([[1]])), axis=0)

    mono_VO = MonocularVisualOdometry(
        intrinsics,
        distortion_coeffs,
        img.shape[0],
        img.shape[1],
        starting_pose,
    )

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Initialize trajectory and ground truth plots
    (trajectory_plot,) = ax.plot([], [], [], label="Camera Path", color="blue")
    (ground_truth_plot,) = ax.plot([], [], [], label="Ground Truth", color="red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim3d(-500, 500)
    ax.set_ylim3d(-500, 500)
    ax.set_zlim3d(-500, 500)

    ax.legend()

    plt.draw()
    plt.pause(0.001)

    for i in range(0, len(images)):
        img = cv.imread(images[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        curr_keypoints, curr_descriptors = mono_VO.compute_keypoints_and_descriptors(
            gray
        )

        if i > 1:
            good_matches, prev_good_keypoints, curr_good_keypoints = (
                mono_VO.compute_matches(
                    prev_descriptors,
                    curr_descriptors,
                    prev_keypoints,
                    curr_keypoints,
                    LOWE_RATIO,
                    1,
                )
            )

            draw_params = dict(
                matchColor=-1, singlePointColor=None, matchesMask=None, flags=2
            )

            # matched_img = cv.drawMatches(
            #     prev_gray,
            #     prev_keypoints,
            #     gray,
            #     curr_keypoints,
            #     good_matches,
            #     None,
            #     **draw_params,
            # )

            # cv.imshow("Matches", matched_img)

            time_step = timestamps[i] - timestamps[i - 1]

            camera_pose = mono_VO.compute_pose(
                prev_good_keypoints, curr_good_keypoints, time_step, 2
            )
            # camera_pose = FINAL_TRANSFORM @ camera_pose
            trajectory.append(camera_pose[:3, 3])

        cv.imshow("Cam_feed", gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        homogeneous_transform_gt = np.concatenate(
            (poses[i], np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
        )
        curr_pose = homogeneous_transform_gt @ ground_truth_pose
        ground_truth.append(curr_pose.flatten()[:3])

        # Update the trajectory plot
        trajectory_np = np.array(trajectory)  # Convert trajectory to a NumPy array
        ground_truth_np = np.array(ground_truth)
        if trajectory_np.shape[0] > 1:  # Ensure there are at least two points to plot
            trajectory_plot.set_data(
                trajectory_np[:, 0], trajectory_np[:, 1]
            )  # X and Y
            trajectory_plot.set_3d_properties(trajectory_np[:, 2])  # Z
            ground_truth_plot.set_data(
                ground_truth_np[:, 0], ground_truth_np[:, 1]
            )  # X and Y for ground truth
            ground_truth_plot.set_3d_properties(
                ground_truth_np[:, 2]
            )  # Z for ground truth

            plt.draw()
            plt.pause(0.001)

        prev_gray = gray
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
