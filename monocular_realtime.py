import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from monocular.monocular import MonocularVisualOdometry


def main():
    INTRINSICS = np.array(
        [
            [166.13029993, 0.0, 319.49999036],
            [0.0, 282.319318, 239.50000072],
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

    LOWE_RATIO = 0.8

    TIME_STEP = 0.01  # time step

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

    trajectory = []

    starting_pose = np.zeros((3, 1))

    mono_VO = MonocularVisualOdometry(
        INTRINSICS,
        DISTORTION_COEFF,
        480,
        640,
        starting_pose,
    )

    while True:
        ret, img = frame.read()
        if not ret:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        curr_keypoints, curr_descriptors = mono_VO.compute_keypoints_and_descriptors(
            gray
        )

        if not first_frame:
            good_matches, prev_good_keypoints, curr_good_keypoints = (
                mono_VO.compute_matches(
                    prev_descriptors,
                    curr_descriptors,
                    prev_keypoints,
                    curr_keypoints,
                    LOWE_RATIO,
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
                prev_good_keypoints, curr_good_keypoints, TIME_STEP
            )

        prev_gray = gray
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors

        cv.imshow("Cam_feed", gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        first_frame = False

        # Update the trajectory plot
        trajectory_np = np.array(trajectory)  # Convert trajectory to a NumPy array
        if trajectory_np.shape[0] > 1:  # Ensure there are at least two points to plot
            trajectory_plot.set_data(
                trajectory_np[:, 0], trajectory_np[:, 1]
            )  # X and Y
            trajectory_plot.set_3d_properties(trajectory_np[:, 2])  # Z
            plt.draw()
            plt.pause(0.001)

    frame.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
