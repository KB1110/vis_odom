import cv2 as cv
import numpy as np
import ssc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        if abs(r[1]) < 1e-6:  # Prevent division by zero or near-zero
            continue
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

INTRINSICS = np.array([[166.13029993, 0.0, 319.49999036],
                        [0.0, 282.319318, 239.50000072],
                        [0.0, 0.0, 1.0]])

OPTIMAL_INTRINSICS = np.array([[144.35171509, 0.0, 312.17998081],
                                [0.0, 241.51057434, 239.00072201],
                                [0.0, 0.0, 1.0]])

DISTORTION_COEFF = np.array([[-2.27010164e-02,  1.10733614e-04,  1.33925610e-05, -1.87437026e-03, -1.09357096e-07]])

ROI = np.array((8, 31, 620, 418)) #x, y, w, h

LOWE_RATIO = 0.6

orb = cv.ORB_create(nfeatures = 500, scaleFactor = 1.1, nlevels = 8, edgeThreshold = 31,
                    firstLevel = 0, WTA_K = 4, scoreType = cv.ORB_HARRIS_SCORE,
                    patchSize = 31)

# Brute Force Matcher
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck = False)

x, y, w, h = ROI

trajectory = []

cam_pose = np.zeros((3, 1))
cam_transformation = np.eye(4)

frame = cv.VideoCapture(0, cv.CAP_V4L2)

first_frame = True

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
trajectory_plot, = ax.plot([], [], [], label='Camera Path')
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
    gray = gray[y:y+h, x:x+w]
    gray = cv.flip(gray, 1)

    if not first_frame:
        next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, np.array(prev_pts, dtype=np.float32), None)
        good_prev = np.array(prev_pts)[status.ravel() == 1]
        good_curr = next_pts[status.ravel() == 1]
        
        if len(good_prev) < 8:
            print("[INFO] Optical Flow failed, fallback to ORB keypoints")  
            keypoints = orb.detect(gray, None)
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse = True)
            selected_keypoints = ssc.ssc(
                keypoints, 300, 0.5, gray.shape[1], gray.shape[0]
            )
            selected_keypoints, selected_descriptors = orb.compute(gray, selected_keypoints)
            matches = bf.knnMatch(prev_descriptors, selected_descriptors, k = 2)
        
            good_prev = []
            good_curr = []
        
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < LOWE_RATIO * n.distance:
                    good_prev.append(prev_keypoints[m.queryIdx].pt)
                    good_curr.append(selected_keypoints[m.trainIdx].pt)
        
            good_prev = np.asarray(good_prev, dtype=np.float32)
            good_curr = np.asarray(good_curr, dtype=np.float32)
            
            prev_descriptors = selected_descriptors
            prev_keypoints = selected_keypoints

        img_display = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        for pt in good_curr:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv.circle(img_display, tuple(map(int, pt)), 4, color, -1)

        cv.imshow("Tracked Points", img_display)
        
        if good_prev.shape[0] < 8 or good_curr.shape[0] < 8:
            print("[WARN] Not enough points for findEssentialMat")
            prev_pts = [kp.pt for kp in selected_keypoints]
            continue
            
        E, mask = cv.findEssentialMat(good_prev, good_curr, OPTIMAL_INTRINSICS, method = cv.RANSAC, prob = 0.999, threshold = 1.0)
        print("Mean flow:", np.mean(np.linalg.norm(good_prev - good_curr, axis=1)))
        print("Inliers:", np.sum(mask), "/", len(mask))
        print("Essential Matrix: ")
        print(E)
        
        flow_magnitude = np.mean(np.linalg.norm(good_prev - good_curr, axis=1))
        if flow_magnitude < 0.8:
            print("Likely no motion — skipping pose update")
            prev_pts = good_curr
        
        if (E is not None and mask is not None and E.shape[0] % 3 == 0) and flow_magnitude >= 0.8:
            if E.shape[0] > 3:
                E = E[:3, :]

            retval, R, t, mask_pose = cv.recoverPose(E, good_prev, good_curr, OPTIMAL_INTRINSICS, mask = mask)
            
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3:] = t
            cam_transformation = cam_transformation @ H
            cam_pose = cam_transformation[:3, 3].reshape(3, 1)
            
            
            trajectory.append(cam_transformation[:3, 3].flatten())
            trajectory_np = np.array(trajectory)
            trajectory_plot.set_data(trajectory_np[:, 0], trajectory_np[:, 1])
            trajectory_plot.set_3d_properties(trajectory_np[:, 2])
            plt.draw()
            plt.pause(0.001)

            # lines1 = cv.computeCorrespondEpilines(good_curr.reshape(-1, 1, 2), 2, E)
            # lines1 = lines1.reshape(-1, 3)
            # img5, img6 = drawlines(prev_gray, gray, lines1, good_prev, good_curr)
            # epilines_img = np.hstack((img5, img6))
            # cv.imshow("Epilines", epilines_img)
                
        prev_pts = good_curr

    else:
        keypoints = orb.detect(gray, None)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse = True)
        selected_keypoints = ssc.ssc(
            keypoints, 300, 0.5, gray.shape[1], gray.shape[0]
        )
        selected_keypoints, selected_descriptors = orb.compute(gray, selected_keypoints)
        prev_keypoints = selected_keypoints
        prev_descriptors = selected_descriptors
        prev_pts = [kp.pt for kp in selected_keypoints]
        first_frame = False

    prev_gray = gray.copy()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv.destroyAllWindows()

print(trajectory)