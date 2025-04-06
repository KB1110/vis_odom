'''
Code taken and modified from:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
'''
import cv2 as cv
import numpy as np
import glob
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(script_dir, 'checkerboard_imgs')

images = glob.glob(os.path.join(images_dir, '*.png'))

pattern_size = (8,6); #interior number of corners

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0 : pattern_size[1], 0 : pattern_size[0]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(10)
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibrated image', dst)

print("Camera matrix: ")
print(mtx)
print("Optimal Camera matrix: ")
print(newcameramtx)
print("Distortion coefficients: ")
print(dist)
print("ROI: ")
print(roi)

cv.waitKey(10000)
