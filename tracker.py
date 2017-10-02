# class that tracks the lane lines using convolution and past lane references.
# Some code used from lecture material.

import numpy as np
import cv2

class Tracker(object):
	'''Class that creates a tracker object to track the lane lines.'''
	def __init__(self, my_window_width, my_window_height, my_margin, my_ym=1, my_xm=1, my_smooth_factor=15):
		# list that stores all the past (left, right) center set values used for smoothing the output
		self.past_centers = []

		# the window pixel width of the center values, used to count pixels inside center windows to determine curve values
		self.window_width = my_window_width

		# the window pixel height of the center values, used to count pixels inside center windows to determine curve values
		# breaks the image into vertical levels
		self.window_height = my_window_height

		# the pixel distance in both directions to slide (left_window + right_window) template for searching
		self.margin = my_margin
		self.ym_per_pixel = my_ym	# meters/pixel in y axis
		self.xm_per_pixel = my_xm	# meters/pixel in x axis
		self.smooth_factor = my_smooth_factor

	def find_lanes(self, warped):
		'''Function that finds the lanes and keeps track.'''
		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin

		centers = []   # store the (left, right) window centroid postions per level
		window = np.ones(window_width) # create our window template that we will use for convolutions

		# Frist find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
		# and then np.convolve the vertical image slice with the window template

		# Sum quarter bottom of image to get slice, could use a different ratio
		left_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		left_center = np.argmax(np.convolve(window,left_sum))-window_width/2
		right_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		right_center = np.argmax(np.convolve(window,right_sum))-window_width/2+int(warped.shape[1]/2)

		# add what we found for the first layer
		centers.append((left_center, right_center))

		# go through each layer looking for max pixel locations
		for level in range(1, (int)(warped.shape[0]/window_height)):
				# convolve the window into the vertical slice of the image
				image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
				convolve_signal = np.convolve(window, image_layer)
				# find the best left centroid by using past left center as a reference
				# use window_width/2 as offset because conv signal reference is at right side of window, not center of window
				offset = window_width/2
				left_min_index = int(max(left_center+offset-margin,0))
				left_max_index = int(min(left_center+offset+margin, warped.shape[1]))
				left_center = np.argmax(convolve_signal[left_min_index:left_max_index]) + left_min_index-offset

				# find the best right centroid by using past right center as a reference
				right_min_index = int(max(right_center+offset-margin,0))
				right_max_index = int(min(right_center+offset+margin, warped.shape[1]))
				right_center = np.argmax(convolve_signal[right_min_index:right_max_index]) + right_min_index-offset
				# add what we found for that layer
				centers.append((left_center, right_center))

		self.past_centers.append(centers)
		# return averaged values of the line centers, helps to keep the markers from jumping around too much
		return np.average(self.past_centers[-self.smooth_factor:], axis = 0)