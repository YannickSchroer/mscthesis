# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import keras.callbacks
import scipy.ndimage.interpolation as trans
import scipy.misc as misc
import PIL

class Distortions(keras.callbacks.Callback):
	def __init__(self, x, y, number_of_images, original_resolution, normalize=True):
		self.original_x = np.copy(x)
		self.x = x
		self.original_y = np.copy(y)
		self.y = y
		self.resolution = x.shape[2:4]
		self.original_resolution = original_resolution
		self.resolution_factor_x = self.original_resolution[0] / self.resolution[0]
		self.resolution_factor_y = self.original_resolution[1] / self.resolution[1]
		self.number_of_images = number_of_images
		self.normalize = normalize

	def repair_image_and_labels(self, image, labels_l, labels_r):
		'''This method is used to edit the transformed images such that they get back the resolution demanded by the network.'''
		# if image broader than res[1], throw away left and right border
		if image.shape[1] > self.resolution[0]:
			left = (image.shape[1] - self.resolution[0]) // 2
			right = left + self.resolution[0]
			image = image[:,left:right,:]
			labels_l = self.shift_label(labels_l, (-left,0))
			labels_r = self.shift_label(labels_r, (-left,0))

		# if image higher than res[2], throw away top and bottom border
		if image.shape[2] > self.resolution[1]:
			top = (image.shape[2] - self.resolution[1]) // 2
			bottom = top + self.resolution[1]
			image = image[:,:,top:bottom]
			labels_l = self.shift_label(labels_l, (0,-top))
			labels_r = self.shift_label(labels_r, (0,-top))

		# if image smaller in x-dim, add pixels to the left and right border
		if image.shape[1] < self.resolution[0]:
			left = (self.resolution[0] - image.shape[1]) // 2
			right = left
			if image.shape[1] % 2 == 1:
				left += 1
			image = np.pad(image, ((0,0),(left,right),(0,0)), 'edge')
			labels_l = self.shift_label(labels_l, (left,0))
			labels_r = self.shift_label(labels_r, (left,0))

		# if image smaller in y-dim, add pixels to the top and bottom border
		if image.shape[2] < self.resolution[1]:
			top = (self.resolution[1] - image.shape[2]) // 2
			bottom = top
			if image.shape[2] % 2 == 1:
				top += 1
			image = np.pad(image, ((0,0),(0,0),(top,bottom)), 'edge')
			labels_l = self.shift_label(labels_l, (0,top))
			labels_r = self.shift_label(labels_r, (0,top))

		return image, labels_l, labels_r

	def rotate_label(self, point, angle):
		'''Rotate a label anti-clockwise by a given angle around the center of the image. The angle is given in degrees.'''
		angle = math.radians(angle)

		ox, oy = int(self.original_resolution[0]) / 2, int(self.original_resolution[1]) / 2
		px, py = point[0], point[1]

		cosine = math.cos(angle)
		sine = math.sin(angle)

		qx = ox + cosine * (px - ox) - sine * (py - oy)
		qy = oy + sine * (px - ox) + cosine * (py - oy)

		return qx, qy

	def shift_label(self, point, translation):
		'''Shift a label by 'translation' * resolutionFactor pixels'''
		return point[0] + translation[0] * self.resolution_factor_x, point[1] + translation[1] * self.resolution_factor_y

	def scale_label(self, point, scale_factors):
		'''Scale a label by the values given in 'scale_factors'''
		return point[0] * scale_factors[0], point[1] * scale_factors[1]

	def on_epoch_begin(self, epoch, logs={}):
		'''This method is called at the beginning of each epoch. Images and labels are shifted, rotated and scaled.'''
		# create random values for shift, rotation and scaling
		shift_values = np.random.uniform(- 0.2 * self.x.shape[1], 0.2 * self.x.shape[2], (self.number_of_images))
		rotate_angles = np.random.uniform(-5., 5., (self.number_of_images))
		scale_factors = np.random.uniform(0.95, 1.05, (self.number_of_images))

		# iterate over all images and transform them
		for img_id in range(self.number_of_images):
			# ~~~~~~~~~~~~~~~~ Transform image ~~~~~~~~~~~~~~~~ #

			# shift (does not change shape)
			img = trans.shift(self.original_x[img_id], [0, shift_values[img_id], shift_values[img_id]], mode="nearest", order=1)

			# rotate (does not change shape)
			img = trans.rotate(img, axes=(1,2), angle=rotate_angles[img_id], reshape=False, mode="nearest", order=1)

			# scale (changes shape) and ignore warnings
			old_stderr = sys.stderr
			nirvana = open(os.devnull, 'w')
			sys.stderr = nirvana
			img = trans.zoom(img, zoom=[1,scale_factors[img_id],scale_factors[img_id]], mode="nearest", order=1)
			sys.stderr = old_stderr

			# ~~~~~~~~~~~~~~~~ Transform labels ~~~~~~~~~~~~~~~ #

			# shift labels			
			labels_l = self.shift_label(self.original_y[img_id][:2], (shift_values[img_id],shift_values[img_id]))
			labels_r = self.shift_label(self.original_y[img_id][2:], (shift_values[img_id],shift_values[img_id]))

			# rotate labels
			labels_l = self.rotate_label(labels_l, rotate_angles[img_id])
			labels_r = self.rotate_label(labels_r, rotate_angles[img_id])
			
			# scale labels
			labels_l = self.scale_label(labels_l, (scale_factors[img_id], scale_factors[img_id]))
			labels_r = self.scale_label(labels_r, (scale_factors[img_id], scale_factors[img_id]))

			self.x[img_id], self.y[img_id][:2], self.y[img_id][2:] = self.repair_image_and_labels(img, labels_l, labels_r)			

	def show_image(self, image): #TODO Remove me
		'''This method opens a window with an image created from a (3*x*y) color image.'''
		if self.normalize:
			foo = PIL.Image.fromarray(np.transpose((image * 255).astype('uint8'), (2,1,0)))
		else:
			foo = PIL.Image.fromarray(np.transpose((image).astype('uint8'), (2,1,0)))
		foo.show()
