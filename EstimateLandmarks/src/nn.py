# -*- coding: utf-8 -*-

import argparse
import json

import theano
theano.config.openmp = True
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import keras.optimizers as optimizers

def load_layout(path):
	'''This method loads a layout in JSON format'''
	with open(path) as in_file:
		layout = json.load(in_file)
	return layout

def store_layout(layout, path):
	'''This method saves a layout as JSON file'''
	with open(path, 'w') as out_file:
		json.dump(layout, out_file, indent=2, sort_keys = True)

def build_model_to_layout(layout, momentum=0.9, nesterov=False):
	'''This method contructs a model according to a given layout.'''
	model = models.Sequential()

	for ltype, lspec in layout:
		if ltype == 'conv2D':
			layer = conv_layers.Convolution2D(**lspec)
		elif ltype == 'maxpool2D':
			layer = conv_layers.MaxPooling2D(**lspec)
		elif ltype == 'flatten':
			layer = core_layers.Flatten()
		elif ltype == 'dense':
			layer = core_layers.Dense(**lspec)
		elif ltype == 'droupout':
			layer = core_layers.Dropout(**lspec)
		else:
			print(ltype) # TODO REMOVE ME
			raise NotImplementedError

		model.add(layer)

	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=momentum, nesterov=nesterov) # TODO Rethink this. Use RMSPROP?
	model.compile(loss='mean_squared_error', optimizer=sgd)

	return model, sgd
