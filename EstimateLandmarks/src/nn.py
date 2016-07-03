# -*- coding: utf-8 -*-

import argparse
import json

import theano
theano.config.openmp = True
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import keras.optimizers as optimizers
import keras.backend as K

from scaled_tanh import scaled_tanh

def load_layout(path):
	'''This method loads a layout in JSON format'''
	with open(path) as in_file:
		layout = json.load(in_file)
	return layout

def store_layout(layout, path):
	'''This method saves a layout as JSON file'''
	with open(path, 'w') as out_file:
		json.dump(layout, out_file, indent=2, sort_keys = True)

def inter_ocular_distance_error(y_true, y_pred):
	# this does not work
	iod = K.sqrt(K.square(y_true[:,1] - y_true[:,0]) + K.square(y_true[:,3] - y_true[:,2]))
	err = y_pred - y_true
	err[:,0] /= iod
	err[:,1] /= iod
	err[:,2] /= iod
	err[:,3] /= iod

	return err

	#exit()
	#return K.mean(K.square(y_pred - y_true), axis=-1)

def build_model_to_layout(layout, learningrate = 0.01, momentum=0.9, decay = 0.0, nesterov=False):
	'''This method contructs a model according to a given layout.'''
	model = models.Sequential()

	for ltype, lspec in layout:
		try:
			if lspec['activation'] == 'scaled_tanh':
				lspec['activation'] = scaled_tanh
		except KeyError:
			pass

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
			raise NotImplementedError

		model.add(layer)

	#optimizer = optimizers.RMSprop(lr=learningrate)
	optimizer = optimizers.SGD(lr=learningrate, momentum=momentum, decay=decay, nesterov=True)
	model.compile(loss='mse', optimizer=optimizer)
	#model.compile(loss=inter_ocular_distance_error, optimizer=optimizer)

	return model, optimizer
