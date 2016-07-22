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
import custom_callbacks
import visualize as vs

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for [default 1]', type=int, default=1)
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-m', '--morph', help='Morph training data between epochs', action='store_true')
parser.add_argument('-n', '--normalize', help='Normalize input data. Choose value 0 for no normalization, value 1 for normalization to the interval [0,1] and value 2 for normalization to the interval[-1,1]', type=int, default=0)
parser.add_argument('-o', '--normalize-output', help='Determine whether the output of the network shall be normalized to the interval [0,1]. The labels are divided by the largest dimension of the input data.', action='store_true')
parser.add_argument('-g', '--grayscale', help='Convert color images to gray scale images.', action='store_true')
parser.add_argument('-w', '--weightscale', help='[-w,+w] is the range for the weights after the initialization. Is not applied if weights are loaded.', type=float, default=1)
parser.add_argument('-r', '--learningrate', help='Learning rate.', type=float, default=0.01)
parser.add_argument('-l', '--load-status', help='Basename of the files to load status from')
parser.add_argument('-s', '--store-status', help='Basename of the files to store status in')
parser.add_argument('-x', '--store-histogram', help='Path where the loss histogram shall be stored to. The path should include the file name, but not the file ending.')
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()

# Load model
print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout, args.learningrate, momentum=0.0, decay=0.001)

# Get input shape and resolution
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]

# Print from where the images are loaded, to which resolution they are scaled and whether they are normalized
if args.normalize == 1:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [0,1]'.format(args.path, resolution[0], resolution[1]))
elif args.normalize == 2:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [-1,1]'.format(args.path, resolution[0], resolution[1]))
else:
	print('Loading data from {0} and rescaling it to {1}x{2}. Images are not normalized!'.format(args.path, resolution[0], resolution[1]))

# Load data
x_train, y_train, original_resolution = dataset_io.read_data(args.path, resolution, args.datalimit, normalize=args.normalize, grayscale=args.grayscale, return_original_resolution=True)
nb_labels = layout[-1][1]['output_dim']
max_dim = np.max(original_resolution)

if args.normalize_output:
	y_train /= max_dim

# Load status
if args.load_status:
	dataset_io.load_status(model, optimizer, args.load_status)
#else: #TODO Construct case distinction for glorot vs uniform initialization
#	if args.weightscale:
#		weights = model.get_weights()
#		helpers.mult_list(weights, args.weightscale * 20)
#		model.set_weights(weights)

#~ Create distortions callback
if args.morph:
	if args.normalize_output:
		callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], (float(original_resolution[0]) / float(max_dim), float(original_resolution[1]) / float(max_dim)))]
	else:
		callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], original_resolution)]
	print('Distortions will be applied to training data between epochs.')
else:
	callbacks = []
loss_callback = custom_callbacks.RecordLoss(args.epochs, nb_labels, x_train, y_train, resolution, model, args.grayscale)
callbacks.append(loss_callback)

# Train the model
print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], args.batchsize, args.epochs))
#for i in range(args.epochs):
model.fit(x_train, y_train, nb_epoch=args.epochs, batch_size=args.batchsize, shuffle=True, callbacks=callbacks, verbose=args.verbosity)
	#print(model.evaluate(x_train, y_train, args.batchsize))

# Calculate loss
#loss, losses = calculate_loss(x_train, resolution, model, grayscale=args.grayscale, return_losses=True)
#print("Loss: ", loss)
print loss_callback.loss_history

# Create histogram
if args.store_histogram:
	vs.loss_histogram(losses, args.store_histogram)

# Store status
if args.store_status:
	dataset_io.store_status(model, optimizer, args.store_status)

print('DONE')
