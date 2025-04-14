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
    pose_file = os.path.join(
        dataset_path, KITTI_POSE_PATH, f"{KITTI_SEQUENCE}.txt"
    )
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
    intrinsics = calibration_matrix[:, :3]
    starting_pose = calibration_matrix[:, 3].reshape(3, 1)
    distortion_coeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0,]])

    return images, poses, timestamps, intrinsics, starting_pose, distortion_coeffs


def main():
    KITTI_DATASET_PATH = "/home/kb/kitti-dataset/"
    KITTI_SEQUENCE = "00"

    images, poses, timestamps, intrinsics, starting_pose, distortion_coeffs = load_kitti_sequence(KITTI_DATASET_PATH, KITTI_SEQUENCE)
    img = cv.imread(images[0])

    trajectory = []
    
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
    (trajectory_plot,) = ax.plot([], [], [], label="Camera Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim3d(-1500, 1500)
    ax.set_ylim3d(-1500, 1500)
    ax.set_zlim3d(-1500, 1500)

    ax.legend()

    plt.draw()
    plt.pause(0.001)
    
    for i in range(0, len(images)):
        img = cv.imread(images[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        curr_keypoints, curr_descriptors = mono_VO.compute_keypoints_and_descriptors(gray)

        if i > 1:
            good_matches, prev_good_keypoints, curr_good_keypoints = (
                mono_VO.compute_matches(
                    prev_descriptors,
                    curr_descriptors,
                    prev_keypoints,
                    curr_keypoints,
                )
            )
            
            draw_params = dict(
                matchColor=-1, singlePointColor=None, matchesMask=None, flags=2
            )

            matched_img = cv.drawMatches(
                prev_gray,
                prev_keypoints,
                gray,
                curr_keypoints,
                good_matches,
                None,
                **draw_params
            )

            cv.imshow("Matches", matched_img)

            trajectory = mono_VO.compute_trajectory(
                prev_good_keypoints, curr_good_keypoints, timestamps[i] - timestamps[i - 1]
            )
            
        cv.imshow("Cam_feed", gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Update the trajectory plot
        trajectory_np = np.array(trajectory)  # Convert trajectory to a NumPy array
        if trajectory_np.shape[0] > 1:  # Ensure there are at least two points to plot
            trajectory_plot.set_data(
                trajectory_np[:, 0], trajectory_np[:, 1]
            )  # X and Y
            trajectory_plot.set_3d_properties(trajectory_np[:, 2])  # Z
            plt.draw()
            plt.pause(0.001)

        prev_gray = gray
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors
    
    cv.destroyAllWindows()
            
if __name__ == "__main__":
    main()
    
