# -*- coding: utf-8 -*-

'''Take a network layout and weights, and test its performance on a given data set.
Print score and accuracy.
User can specify batchsize, number of samples to consider (randomly chosen), and keras verbosity'''

import argparse

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils
import numpy as np

import nn
import dataset_io
import visualize

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('weights', help='Path weights in xyz file')
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
parser.add_argument('-p', '--predsave', help='The path to which the images with predicted land marks shall be saved', type=str, default=None)
parser.add_argument('-s', '--predshow', help='Defines whether predictions shall be displayed. It is highly recommended to choose this option only with an appropriate choice of d.', action='store_true')
parser.add_argument('-c', '--crosssize', help='Set the size of the drawn crosses', type=int, default=2)
args = parser.parse_args()

# Load model
print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout)

# Load weights
print('Loading weights from {0}'.format(args.weights))
model.load_weights('{0}.w'.format(args.weights))

# Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}'.format(args.path, resolution[0], resolution[1]))
x_pred, y_true, image_list = dataset_io.read_data(args.path, resolution, args.datalimit, labels=True, return_image_properties=True)

# Predict
print('Predict on {0} samples at resolution {1}x{2} in batches of size {3}'.format(x_pred.shape[0], resolution[0], resolution[1], args.batchsize))
predictions = model.predict(x_pred, batch_size = args.batchsize, verbose=args.verbosity) * 640

# Concatenate true labels and predictions
y_pred_and_true = np.concatenate((y_true, predictions), axis=1)

# Save and/or display predictions
if args.predsave != None and args.predshow:
	# Save and show predictions
	print('Save images with drawn predicted land marks to path \'{0}\'. The images will be displayed.'.format(args.predsave))
	visualize.visualize_predictions(image_list, predictions, y_true, args.crosssize, args.predsave, args.predshow)
elif args.predshow:
	# Only show predictions
	print('Display images with drawn predicted land marks.')
	visualize.visualize_predictions(image_list, predictions, y_true, args.crosssize, args.predsave, args.predshow)
elif args.predsave != None:
	# Only save predictions
	print('Save images with drawn predicted land marks to path \'{0}\'.'.format(args.predsave))
	visualize.visualize_predictions(image_list, predictions, y_true, args.crosssize, args.predsave, args.predshow)

print('DONE')
