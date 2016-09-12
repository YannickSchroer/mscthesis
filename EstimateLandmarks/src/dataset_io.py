# -*- coding: utf-8 -*-

import pickle
import PIL.Image as pil
import PIL.ImageOps as pilops
import numpy as np
import time
import random
import csv

def create_image_list(csv_path, labels=True):
	'''This method takes the path to a CSV file and saves all images in a list. If the parameter 'labels' is 'False', the labels will be ignored.'''
	
	# create list, which saves a dictionary for each image
	image_list = []

	# open CSV file to read images
	with open(csv_path) as csv_file:
		# create CSV reader
		csv_reader = csv.reader(csv_file, delimiter=";")

		# get original image resolution
		next(csv_reader)
		for image_row in csv_reader:
			original_resolution = (int(image_row[1]), int(image_row[2]))
			break

		# go back to the begin of the file
		csv_file.seek(0)
		next(csv_reader)

		# iterate over images
		for image_row in csv_reader:
			# append image to image list
			if labels:
				image_list.append({'path': image_row[0], 'labels': image_row[3:]})
			else:
				image_list.append({'path': image_row[0]})

	#shuffle images
	random.shuffle(image_list)

	return image_list, original_resolution

def read_data(csv_path, resolution, d=None, normalize=2, autocontrast=True, grayscale=False, return_image_properties=False, return_original_resolution=False, labels=True):
	'''In order to construct the data structures, which contain the training and test data, this method takes the path to a csv file, which saves information about the images. In addition, this method takes the 'resolution' to which the images are to be scaled, a parameter 'd' which determines how many images are to be processed, an int parameter 'normalize' which controls whether the images shall be normalized to the interval [0,1] (noramlize=1) or [-1,1] (normalize=2) or not at all (normalize=0) and a boolean parameter 'autocontrast' which controls whether a PIL intern method shall be used to increase the contrast of the images. The parameter 'grayscale' determines whether the color information will be used or not. The boolean parameter 'return_image_properties' can be set to 'True' in order to return the image list created by the create_image_list method called within this method. If the flag 'labels' is set to 'False', the labels will be ignored'''
	#create image list
	image_list, original_resolution = create_image_list(csv_path, labels)

	# check whether there is a limit for the images to be loaded
	num_images = d if d is not None else len(image_list)

	# throw away everything behind the first d elements
	if d != len(image_list):
		image_list = image_list[:d]

	# create empty arrays with appropriate size
	if grayscale:
		X = np.empty((num_images, 1, resolution[0], resolution[1]), dtype=float)
	else:
		X = np.empty((num_images, 3, resolution[0], resolution[1]), dtype=float)
	if labels:
		y = np.empty((num_images, len(image_list[0]['labels'])))

	# iterate over images
	for idx, image in enumerate(image_list):
		if idx >= num_images:
			break

		# open the image
		if grayscale:
			im = pil.open(image['path']).convert('L')
		else:
			im = pil.open(image['path'])

		# autocontrast
		if autocontrast:
			im = pilops.autocontrast(im)

		# resize image to desired size
		im = im.resize(resolution)

		# save image as array within the result array
		if grayscale:
			X[idx] = np.transpose(np.asarray(im), [1,0])
		else:
			X[idx] = np.transpose(np.asarray(im), [2,1,0])

		if labels:
			# save target values
			y[idx] = image['labels']

	# normalize images to range [0,1] or [-1,1] if desired
	if normalize > 0:
		X /= 255.0
		if normalize == 2:
			X *= 2
			X -= 1

	if labels:
		if return_image_properties:
			if return_original_resolution:
				return X, y, image_list, original_resolution			
			else:
				return X, y, image_list
		else:
			if return_original_resolution:
				return X, y, original_resolution
			else:
				return X, y
	else:
		if return_image_properties:
			if return_original_resolution:
				return X, image_list, original_resolution
			else:
				return X, image_list
		else:
			if return_original_resolution:
				return X, original_resolution
			else:
				return X

def load_status(model, optimizer, filename, verbose=True):
	'''This method loads the state of the model and of the optimizer from the file system'''
	if verbose:
		print('Loading status from {0}'.format(filename))

	# weights
	weight_filename = filename + ".w"
	model.load_weights(weight_filename)

	# training parameters
	try:
		train_filename = filename + ".t"
		with open(train_filename, 'rb') as train_file:
			params = pickle.load(train_file)
		optimizer.updates = params[1]
		optimizer.set_state(params[0])
	except IOError:
		print('\t{0} not found: using initial parameters'.format(train_filename))

def store_status(model, optimizer, filename, verbose=True):
	'''This method stores the state of the model and of the optimizer to the file system'''
	if verbose:	
		print('Storing status to {0}'.format(filename))

	# weights
	weight_filename = filename + ".w"
	model.save_weights(weight_filename, overwrite=True)

	# training parameters
	train_filename = filename + ".t"
	optimizer_state = optimizer.get_state()
	with open(train_filename, 'wb') as train_file:
		pickle.dump([optimizer_state, optimizer.updates], train_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	X, y = read_data("data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv", (48,68))
