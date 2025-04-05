import cv2 as cv
import numpy as np
import ssc

orb = cv.ORB_create(nfeatures = 500, scaleFactor = 1.1, nlevels = 8, edgeThreshold = 31,
                    firstLevel = 0, WTA_K = 2, scoreType = cv.ORB_HARRIS_SCORE,
                    patchSize = 31)

frame = cv.VideoCapture(0, cv.CAP_V4L2)

width = frame.get(cv.CAP_PROP_FRAME_WIDTH)
height = frame.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"Width: {width}, Height: {height}")

intrinsics = np.array([[1.0, 0, width / 2],
                       [0, 1.0, height / 2],
                       [0, 0, 1.0]])

while True:
    ret, img = frame.read()
    if not ret:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.undistort(gray, intrinsics)
    gray = cv.flip(gray, 1)
    keypoints = orb.detect(gray, None)

    keypoints = sorted(keypoints, key=lambda x: x.response, reverse = True)

    selected_keypoints = ssc.ssc(
        keypoints, 300, 2.5, gray.shape[1], gray.shape[0]
    )
    selected_descriptions = orb.compute(gray, selected_keypoints)

    img = cv.drawKeypoints(gray, selected_keypoints, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("After", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv.destroyAllWindows()