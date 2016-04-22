# -*- coding: utf-8 -*-

'''Take a network layout and weights, and test its performance on a given data set.
Print score and accuracy.
User can specify batchsize, number of samples to consider (randomly chosen), and keras verbosity'''

import argparse

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils

import nn
import dataset_io

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('weights', help='Path weights in xyz file')
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
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
x_test, y_test = dataset_io.read_data(args.path, resolution, args.datalimit)

# Test the model
print('Testing on {0} samples at resolution {1}x{2} in batches of size {3}'.format(x_test.shape[0], resolution[0], resolution[1], args.batchsize))
score = model.evaluate(x_test, y_test, batch_size = args.batchsize, verbose=args.verbosity)
print('Test score:', score)

print('DONE')
