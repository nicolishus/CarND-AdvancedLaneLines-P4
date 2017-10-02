# file that calibrates the camera distortion and saves the coefficients into a pickle
# file. Some code used from lecture material.

import numpy as np
import cv2
import glob
import pickle

# prepare object points for distortion correction
# this is needed by the cv2 undistortion function
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points
object_points = []
image_points = []

# Make list of calibratiion images
images = glob.glob("./camera_cal/calibration*.jpg")

# Step through the list and seach for chessboard corners
for i, filename in enumerate(images):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

	# If found, add object points and image points
	if ret == True:
		print ("Working on,", filename)
		object_points.append(objp)
		image_points.append(corners)

		# Draw and display the corners
		cv2.drawChessboardCorners(img, (9,6), corners, ret)
		write_name = "corners_found" + str(i) + ".jpg"
		cv2.imwrite("./camera_cal/" + write_name, img)

# Load image for reference
image = cv2.imread("./camera_cal/calibration1.jpg")
image_size = (image.shape[1], image.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, \
												  image_size, None, None)

# saves the first calibration image after undistorting
distorted_image = cv2.imread("./camera_cal/calibration1.jpg")
undistorted_image = cv2.undistort(distorted_image, mtx, dist, None, mtx)
write_name = "calibration1_undistorted.jpg"
cv2.imwrite("./camera_cal/" + write_name, undistorted_image)

# saves the first test image after undistorting
distorted_image = cv2.imread("./test_images/test1.jpg")
undistorted_image = cv2.undistort(distorted_image, mtx, dist, None, mtx)
write_name = "test1_undistorted.jpg"
cv2.imwrite("./test_images/" + write_name, undistorted_image)

# Save the camera calibration result for later use

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./camera_cal/calibration_pickle.p", "wb"))