# -*- coding: utf-8 -*-

import argparse
import pickle #TODO Remove me

import theano
theano.config.openmp = False
import keras.utils.np_utils as np_utils
import keras.callbacks
import numpy as np

import nn
import dataset_io
import helpers
import distortions

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for [default 1]', type=int, default=1)
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-m', '--morph', help='Morph training data between epochs', action='store_true')
parser.add_argument('-n', '--normalize', help='Normalize input data to values in the interval [0,1]', action='store_true')
parser.add_argument('-w', '--weightscale', help='[-w,+w] is the range for the weights after the initialization. Is not applied if weights are loaded.', type=float, default=1)
parser.add_argument('-r', '--learningrate', help='Learning rate.', type=float, default=0.01)
parser.add_argument('-l', '--load-status', help='Basename of the files to load status from')
parser.add_argument('-s', '--store-status', help='Basename of the files to store status in')
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()

print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout, args.learningrate)

# Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}. Normalize input images to [0,1]: {3}'.format(args.path, resolution[0], resolution[1], args.normalize))
x_train, y_train, original_resolution = dataset_io.read_data(args.path, resolution, args.datalimit, normalize=args.normalize, return_original_resolution=True)

# normalize labels
if False:
	y_train[:,0] /= 480
	y_train[:,1] /= 640
	y_train[:,2] /= 480
	y_train[:,3] /= 640

# Load status
if args.load_status:
	dataset_io.load_status(model, optimizer, args.load_status)
else:
	if args.weightscale:
		weights = model.get_weights()
		helpers.mult_list(weights, args.weightscale * 20)
		model.set_weights(weights)

#~ Create distortions callback
if args.morph:
	callbacks = [distortions.Distortions(x_train, y_train, x_train.shape[0], original_resolution)]
	print('Distortions will be applied to training data between epochs.')
else:
	callbacks = []

# Train the model
print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], args.batchsize, args.epochs))
for i in range(args.epochs):
	model.fit(x_train, y_train, nb_epoch=1, batch_size=args.batchsize, shuffle=False, callbacks=callbacks, verbose=args.verbosity)
	print(model.evaluate(x_train, y_train, args.batchsize))

#errors = []
#for idx, x in enumerate(x_train):
#	x.resize(1,3,96,128)
#	y = np.empty((1,4))
#	y[0] = y_train[idx]
#	errors.append(model.evaluate(x, y, 1, verbose=False))

print(np.mean(errors))

# Store status
if args.store_status:
	dataset_io.store_status(model, optimizer, args.store_status)

print('DONE')
