# -*- coding: utf-8 -*-

import argparse
import pickle

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils

import nn
import dataset_io

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for [default 1]', type=int, default=1)
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-l', '--load-status', help='Basename of the files to load status from')
parser.add_argument('-s', '--store-status', help='Basename of the files to store status in')
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()

print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout)

# Load status
if args.load_status:
	print('Loading status from {0}'.format(args.load_status))

	# weights
	weight_filename = args.load_status + ".w"
	model.load_weights(weight_filename)

	# training parameters
	try:
		train_filename = args.load_status + ".t"
		with open(train_filename, 'rb') as train_file:
			optimizer_state = pickle.load(train_file)
		optimizer.set_state(optimizer_state)
	except IOError:
		print('\t{0} not found: using initial parameters'.format(train_filename))

# Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}'.format(args.path, resolution[0], resolution[1]))
x_train, y_train = dataset_io.read_data(args.path, resolution, args.datalimit)

# Train the model
print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], args.batchsize, args.epochs))
model.fit(x_train, y_train, nb_epoch=args.epochs, batch_size=args.batchsize, verbose=args.verbosity)

# Store status
if args.store_status:
	print('Storing status to {0}'.format(args.store_status))

	# weights
	weight_filename = args.store_status + ".w"
	movel.save_weights(weight_filename, overwrite=True)

	# training parameters
	train_filename = args.store_status + ".t"
	optimizer_state = optimizer.get_state()
	with open(train_filename, 'wb') as train_file:
		pickle.dump(optimizer_state, train_file, pickle.HIGHEST_PROTOCOL)

print('DONE')
