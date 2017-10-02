# file that processes a video to show the lane lines
# Some code used from lecture material.

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import Tracker

# create global tracker object
window_width = 25
window_height = 80
tracker = Tracker(my_window_width=window_width, my_window_height=window_height, my_margin=25, my_ym=10/720, 
				  my_xm=4/384, my_smooth_factor=15)

# Read tin the saved object_points and image_points
dist_pickle = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def gaussian_blur(img, kernelSize):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernelSize, kernelSize), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channelCount = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignoreMaskColor = (255,) * channelCount
    else:
        ignoreMaskColor = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignoreMaskColor)
    
    #returning the image only where mask pixels are nonzero
    maskedImage = cv2.bitwise_and(img, mask)
    return maskedImage

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def color_threshold(image, sthresh=(0, 255), vthresh=(0,255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

	output = np.zeros_like(s_channel)
	output[(s_binary == 1) & (v_binary == 1)] = 1
	return output

def window_mask(width, height, img_ref, center, level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0] - (level+1)*height):int(img_ref.shape[0]-level*height),
		   max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
	return output

def pre_process_image(image):
	'''Function that runs through all the pre-processing stages before the Perspective transform.'''

	# sobel thresholding
	preprocess_image = np.zeros_like(image[:,:,0])
	gradx = abs_sobel_thresh(image, orient='x', thresh=(12,255))
	grady = abs_sobel_thresh(image, orient='y', thresh=(25, 255))

	# color thresholding
	c_binary = color_threshold(image, sthresh=(100,266), vthresh=(50, 255))
	preprocess_image[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

	# apply region mask
	bottomLeft = (image.shape[1] * .20, image.shape[0] * .9)
	topLeft = (.50 * image.shape[1], .55 * image.shape[0])
	topRight = (.52 * image.shape[1], .55 * image.shape[0])
	bottomRight = (image.shape[1] * .85, image.shape[0] * .9)
	vertices = np.array([[bottomLeft, topLeft, topRight, bottomRight]], dtype=np.int32)
	regioned = region_of_interest(preprocess_image, vertices)

	# apply guassian blur filter
	blurred = gaussian_blur(regioned, 3)

	# return preprocessed image
	return blurred

def perspective_transform(image, blurred):
	'''Function that preforms the perspective transform.'''
	image_size = (image.shape[1], image.shape[0])
	bottom_width = .76
	middle_width = .10
	height_percent = .62
	bottom_trim = .935
	src = np.float32([
		[image.shape[1] * (.5 - middle_width/2), image.shape[0]*height_percent], 
		[image.shape[1] * (.5 + middle_width/2), image.shape[0]*height_percent], 
		[image.shape[1] * (.5 + bottom_width/2), image.shape[0]*bottom_trim],
		[image.shape[1] * (.5 - bottom_width/2), image.shape[0]*bottom_trim]])
	offset = image_size[0] * .25
	dst = np.float32([
		[offset, 0], [image_size[0]-offset, 0], 
		[image_size[0]- offset, image_size[1]], 
		[offset, image_size[1]]])

	# preform the transfrom
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(blurred, M, image_size, flags=cv2.INTER_LINEAR)
	return warped, M, Minv

def draw_resulting_image(image, left_lane, right_lane, inner_lane, M, Minv):
	'''Function that draws the lane lines and background onto the original image.'''
	image_size = (image.shape[1], image.shape[0])
	road = np.zeros_like(image)
	road_background = np.zeros_like(image)
	cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
	cv2.fillPoly(road, [right_lane], color = [0, 0, 255])
	cv2.fillPoly(road, [inner_lane], color = [0, 255, 0])
	cv2.fillPoly(road_background, [left_lane], color=[255,255,255])
	cv2.fillPoly(road_background, [right_lane], color = [255,255,255])

	road_warped = cv2.warpPerspective(road, Minv, image_size, flags=cv2.INTER_LINEAR)
	road_warped_background = cv2.warpPerspective(road_background, Minv, image_size, flags=cv2.INTER_LINEAR)

	# calculate the offset of the car on the road
	base = cv2.addWeighted(image, 1.0, road_warped_background, -1.0, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 0.2, 0.0)
	return result

def metrics(warped, image, yvals, res_yvals, left_point, right_fitx, left_fitx, tracker):
	'''Function that calculates and draws the metrics text on the image (radius and center offset).'''
	ym_per_pixel = tracker.ym_per_pixel	# meters per pixel in y direction
	xm_per_pixel = tracker.xm_per_pixel	# meters per pixel in x direction

	curvature = np.polyfit(np.array(res_yvals, np.float32) *ym_per_pixel, np.array(left_point, np.float32)*xm_per_pixel, 2)
	radius = ((1 + (2*curvature[0]*yvals[-1]*ym_per_pixel + curvature[1])**2)**1.5)/np.absolute(2*curvature[0])

	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_offset = (camera_center-warped.shape[1]/2)*xm_per_pixel
	side_position = "left"
	if center_offset <= 0:
		side_position = "right"

	# draw the text showing curvature, offset, and speed
	cv2.putText(image, "Radius of Curvature = " + str(round(radius,3))+"(m)",(400,50), cv2.FONT_HERSHEY_SIMPLEX, 
				1, (255,255,255),2)
	cv2.putText(image, "Vehicle is "+str(abs(round(center_offset,3)))+"m "+side_position+" of center",(400,100), 
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
	return image


def process_image(img):
	# first undistort image using distortion coefficients from running calibration
	global tracker
	image = cv2.undistort(img, mtx, dist, None, mtx)
	image_size = (image.shape[1], image.shape[0])

	# run all the preprocessing steps (sobel, color, region, gaussian)
	blurred = pre_process_image(image)

	# perform perspective transform
	warped, M, Minv = perspective_transform(image, blurred)

	# Create Tracker object (10m = 720 pixel, and 4m = 384 pixels)
	lane_markers = tracker.find_lanes(warped)

	# points used to draw all the left and right windows
	left_points = np.zeros_like(warped)
	right_points = np.zeros_like(warped)

	# right and left points
	right_point = []
	left_point = []

	for level in range(0, len(lane_markers)):
		# window_mask is a function to draw window areas
		# add center value found in frame to the list of lane points per left, right
		left_point.append(lane_markers[level][0])
		right_point.append(lane_markers[level][1])
		left_mask = window_mask(window_width, window_height, warped, lane_markers[level][0], level)
		right_mask = window_mask(window_width, window_height, warped, lane_markers[level][1], level)

		# add graphic points from window mask here to total pixels found
		left_points[(left_points == 255) | ((left_mask == 1))] = 255
		right_points[(right_points == 255) | ((right_mask == 1))] = 255

	# used for debugging to see how well the lane tracker was preforming
	'''# draw the results 
	template = np.array(right_points+left_points, np.uint8)	# add both left and right window pixels together
	zero_channel = np.zeros_like(template)	# create a zero color channel
	template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)	# make window pixels green
	warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)	# making the original road pixels 3 color channels
	boxed = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)	# overlay the original road image with window results'''

	# fit the lane boundaries to the left, right, center positions found
	yvals = range(0, warped.shape[0])
	res_yvals = np.arange(warped.shape[0] - (window_height/2),0,-window_height)
	
	left_fit = np.polyfit(res_yvals, left_point, 2)
	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
	left_fitx = np.array(left_fitx, np.int32)

	right_fit = np.polyfit(res_yvals, right_point, 2)
	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
	right_fitx = np.array(right_fitx, np.int32)

	left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), 
						np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0), 
						np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2), axis=0), 
						np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

	result = draw_resulting_image(image, left_lane, right_lane, inner_lane, M, Minv)
	result = metrics(warped, result, yvals, res_yvals, left_point, right_fitx, left_fitx, tracker)

	return result

input_video_filename = "project_video.mp4"
output_video_filename = "output_video.mp4"

input_video = VideoFileClip(input_video_filename)
output_video = input_video.fl_image(process_image)
output_video.write_videofile(output_video_filename, audio=False)