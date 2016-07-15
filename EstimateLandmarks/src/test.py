# -*- coding: utf-8 -*-

'''Take a network layout and weights, and test its performance on a given data set.
Print score and accuracy.
User can specify batchsize, number of samples to consider (randomly chosen), and keras verbosity'''

import argparse
import os
import glob

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils

import nn
import dataset_io
import visualize

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('weights', help='Path to the weigths. If the path points to a file, only this single file will be tested. If it points to a folder, all weight files in that folder will be tested.')
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
parser.add_argument('-s', '--savepath', help='Path to which the results shall be saved.')
args = parser.parse_args()

# Load model
print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout)

# Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}'.format(args.path, resolution[0], resolution[1]))
x_test, y_test = dataset_io.read_data(args.path, resolution, args.datalimit)

# normalize labels
if True:
	y_test[:,0] /= 640#480
	y_test[:,1] /= 640
	y_test[:,2] /= 640#480
	y_test[:,3] /= 640

# create list with the pathes to all weights
weight_list = []

# load weight files
if os.path.isfile(args.weights):
	weight_list.append(args.weights)
elif os.path.exists(args.weights):
	weight_path = args.weights if args.weights[-1] == "/" else args.weights + "/"
	weight_list = glob.glob(weight_path + "*.w")
	weight_list.sort()
else:
	raise Exception #TODO Clarify Exception

# initialize score list
score_list = []

# iterate over every weight file
for weights in weight_list:
	# Load weights
	print('Loading weights from {0}'.format(weights))
	model.load_weights('{0}'.format(weights))

	# Test the model
	print('Testing on {0} samples at resolution {1}x{2} in batches of size {3}'.format(x_test.shape[0], resolution[0], resolution[1], args.batchsize))
	score = model.evaluate(x_test, y_test, batch_size = args.batchsize, verbose=args.verbosity)
	print('Test score:', score)



if args.savepath:
	f = open(args.savepath, "w")
	for s in score_list:
		f.write(str(s) + "\n")

print('DONE')
