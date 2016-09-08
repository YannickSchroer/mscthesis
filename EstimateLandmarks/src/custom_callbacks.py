# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import keras.callbacks
import scipy.ndimage.interpolation as trans
import scipy.misc as misc
import PIL

class RecordLoss(keras.callbacks.Callback):
	'''This class is a callback, which computes and saves the loss after each epoch'''
	def __init__(self, nb_epochs, nb_labels, data, labels, resolution, model, grayscale):
		self.nb_epochs = nb_epochs
		self.nb_labels = nb_labels
		self.data = np.copy(data)
		self.labels = np.copy(labels)
		self.resolution = resolution
		self.model = model
		self.grayscale = grayscale
		self.loss_history = np.empty((self.nb_epochs))
		self.latest_losses = np.empty((self.data.shape[0]))

	def calculate_loss(self):
		'''This method calculates the mean loss and the individual losses.'''
		#print('\nCalculating loss ...')
		losses = np.empty((self.data.shape[0]))
		for idx, x in enumerate(np.copy(self.data)):
			if self.grayscale:
				x.resize(1,1,self.resolution[0],self.resolution[1])
			else:
				x.resize(1,3,self.resolution[0],self.resolution[1])
			y = np.empty((1,self.nb_labels))
			y[0] = self.labels[idx]
			losses[idx] = self.model.evaluate(x, y, 1, verbose=False)
		return np.mean(losses), losses

	def on_epoch_end(self, epoch, logs={}):
		self.loss_history[epoch], self.latest_losses = self.calculate_loss()

class RecordLossGabor(keras.callbacks.Callback):
	'''This class is a callback, which computes and saves the loss after each epoch'''
	def __init__(self, nb_epochs, nb_labels, data, labels, resolution, model):
		self.nb_epochs = nb_epochs
		self.nb_labels = nb_labels
		self.data = np.copy(data)
		self.labels = np.copy(labels)
		self.resolution = resolution
		self.model = model
		self.loss_history = np.empty((self.nb_epochs))

	def calculate_loss(self):
		'''This method calculates the mean loss and the individual losses.'''
		#print('\nCalculating loss ...')
		return self.model.evaluate(self.data, self.labels, 1, verbose=False)

	def on_epoch_end(self, epoch, logs={}):
		self.loss_history[epoch] = self.calculate_loss()
		
class StopEarly(keras.callbacks.Callback):
	'''This class is a callback, which ends training, if the loss reaches a certain threshold.'''
	def __init__(self, loss_callback = None, limit = 100000):
		self.loss_callback = loss_callback
		self.limit = limit

	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('loss') < self.limit) == False:
			self.model.stop_training = True
			print("\nStopping early because loss got over " + str(self.limit))

class Distortions(keras.callbacks.Callback):
	'''This class is a callback, which transforms the input images (and labels) before each epoch.'''
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

	def repair_image_and_labels(self, image, labels):
		'''This method is used to edit the transformed images such that they get back the resolution demanded by the network.'''
		# if image broader than res[1], throw away left and right border
		if image.shape[1] > self.resolution[0]:
			left = (image.shape[1] - self.resolution[0]) // 2
			right = left + self.resolution[0]
			image = image[:,left:right,:]
			for l in range(labels.shape[0] / 2):
				labels[2*l:2*l+2] = self.shift_label(labels[2*l:2*l+2], (-left,0))

		# if image higher than res[2], throw away top and bottom border
		if image.shape[2] > self.resolution[1]:
			top = (image.shape[2] - self.resolution[1]) // 2
			bottom = top + self.resolution[1]
			image = image[:,:,top:bottom]
			for l in range(labels.shape[0] / 2):
				labels[2*l:2*l+2] = self.shift_label(labels[2*l:2*l+2], (0,-top))

		# if image smaller in x-dim, add pixels to the left and right border
		if image.shape[1] < self.resolution[0]:
			left = (self.resolution[0] - image.shape[1]) // 2
			right = left
			if image.shape[1] % 2 == 1:
				left += 1
			image = np.pad(image, ((0,0),(left,right),(0,0)), 'edge')
			for l in range(labels.shape[0] / 2):
				labels[2*l:2*l+2] = self.shift_label(labels[2*l:2*l+2], (left,0))

		# if image smaller in y-dim, add pixels to the top and bottom border
		if image.shape[2] < self.resolution[1]:
			top = (self.resolution[1] - image.shape[2]) // 2
			bottom = top
			if image.shape[2] % 2 == 1:
				top += 1
			image = np.pad(image, ((0,0),(0,0),(top,bottom)), 'edge')
			for l in range(labels.shape[0] / 2):
				labels[2*l:2*l+2] = self.shift_label(labels[2*l:2*l+2], (0,top))

		return image, labels

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
		shift_values = np.random.uniform(- 0.1 * self.x.shape[1], 0.1 * self.x.shape[2], (self.number_of_images))
		rotate_angles = np.random.uniform(-5., 5., (self.number_of_images))
		scale_factors = np.random.uniform(0.9, 1.1, (self.number_of_images))

		# iterate over all images and transform them
		for img_id in range(self.number_of_images):
			# ~~~~~~~~~~~~~~~~ Transform image ~~~~~~~~~~~~~~~~ #

			# shift (does not change shape)
			img = trans.shift(self.original_x[img_id], [0, shift_values[img_id], shift_values[img_id]], mode="nearest", order=1)

			# rotate (does not change shape)
			img = trans.rotate(img, axes=(1,2), angle=rotate_angles[img_id], reshape=False, mode="nearest", order=1)

			# scale (changes shape) and ignore warnings that the image shape has changed
			old_stderr = sys.stderr
			nirvana = open(os.devnull, 'w')
			sys.stderr = nirvana
			img = trans.zoom(img, zoom=[1,scale_factors[img_id],scale_factors[img_id]], mode="nearest", order=1)
			sys.stderr = old_stderr

			# ~~~~~~~~~~~~~~~~ Transform labels ~~~~~~~~~~~~~~~ #
			
			labels = np.empty(self.y[img_id].shape)
			for l in range(labels.shape[0] / 2):
				# shift labels
				labels[2*l:2*l+2] = self.shift_label(self.original_y[img_id][2*l:2*l+2], (shift_values[img_id],shift_values[img_id]))

				# rotate labels
				labels[2*l:2*l+2] = self.rotate_label(labels[2*l:2*l+2], rotate_angles[img_id])

				# scale labels
				labels[2*l:2*l+2] = self.scale_label(labels[2*l:2*l+2], (scale_factors[img_id], scale_factors[img_id]))

			self.x[img_id], self.y[img_id] = self.repair_image_and_labels(img, labels)

	def show_image(self, image):
		'''This method opens a window with an image created from a (3*x*y) color image.'''
		if self.normalize:
			foo = PIL.Image.fromarray(np.transpose((image * 255).astype('uint8'), (2,1,0)))
		else:
			foo = PIL.Image.fromarray(np.transpose((image).astype('uint8'), (2,1,0)))
		foo.show()
