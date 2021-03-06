# -*- coding: utf-8 -*-

import argparse
import json

import numpy as np
import theano
theano.config.openmp = True
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import custom_layers
import keras.optimizers as optimizers
import keras.backend as K
import theano as T

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
		elif ltype == 'dropout':
			layer = core_layers.Dropout(**lspec)
		else:
			raise NotImplementedError

		model.add(layer)

	#optimizer = optimizers.RMSprop(lr=learningrate)
	optimizer = optimizers.SGD(lr=learningrate, momentum=momentum, decay=decay, nesterov=True)
	model.compile(loss='mse', optimizer=optimizer)
	#model.compile(loss=inter_ocular_distance_error, optimizer=optimizer)

	return model, optimizer

def build_cnn_model(input_shape=(3,96,128), learningrate = 0.1, momentum=0., decay = 0.0, nesterov=False, initialization="glorot_normal", activation_function="sigmoid", nb_max_pooling=3):
	cnn_model = models.Sequential()
	
	# add convolution2D and maxpooling2D layers
	cnn_model.add(conv_layers.Convolution2D(input_shape = input_shape, activation="relu", init=initialization, nb_filter=32, nb_col=3, nb_row=3))
	if nb_max_pooling > 0:
		cnn_model.add(conv_layers.MaxPooling2D(pool_size=(2, 2)))
	cnn_model.add(conv_layers.Convolution2D(activation="relu", init=initialization, nb_filter=64, nb_col=2, nb_row=2))
	if nb_max_pooling > 1:
		cnn_model.add(conv_layers.MaxPooling2D(pool_size=(2, 2)))
	cnn_model.add(conv_layers.Convolution2D(activation="relu", init=initialization, nb_filter=128, nb_col=2, nb_row=2))
	if nb_max_pooling > 2:
		cnn_model.add(conv_layers.MaxPooling2D(pool_size=(2, 2)))

	# flatten model
	cnn_model.add(core_layers.Flatten())
	
	# add fully connected layers
	cnn_model.add(core_layers.Dense(activation=activation_function, init=initialization, output_dim=300))
	cnn_model.add(core_layers.Dense(activation=activation_function, init=initialization, output_dim=200))

	# add output layer
	cnn_model.add(core_layers.Dense(activation="linear", init=initialization, output_dim=30))

	optimizer = optimizers.SGD(lr=learningrate, momentum=momentum, decay=decay, nesterov=nesterov)
	cnn_model.compile(loss='mse', optimizer=optimizer)

	return cnn_model, optimizer


def build_gabor_model(gabor_filters, input_shape=(3,96,128), learningrate = 0.01, momentum=0., decay = 0.0, nesterov=False, mode="abs", add_conv2=True):
	'''This method constructs a model with gabor convolutional layers and subsequent fully connected layers'''

	# initialize lists to save the single models
	first_layer_models = []

	# iterate over the lists containg all the filters for one value of m
	for m_filters in gabor_filters:
		# get filter shape
		filter_shape = m_filters[0].shape
		nb_row = filter_shape[0]
		nb_col = filter_shape[1]

		# initalize lists, which save all real and imaginary parts for the corresponding m
		real_filters = np.empty((len(m_filters), input_shape[0], filter_shape[0], filter_shape[1]))
		imag_filters = np.empty((len(m_filters), input_shape[0], filter_shape[0], filter_shape[1]))

		# iterate over all L filters for corresponding m
		for idx, fil in enumerate(m_filters):
			# add real and imaginary parts to the corresponding list
			real_filters[idx,:] = np.real(fil)
			imag_filters[idx,:] = np.imag(fil)

		# create real and imaginary model
		real_model = models.Sequential()
		imag_model = models.Sequential()

		train_gabor = False  # train whole network
		#train_gabor = False # train only the layers after the gabor layer

		# add convolution layers
		real_model.add(conv_layers.Convolution2D(activation='relu', trainable=train_gabor, input_shape=input_shape, nb_filter = len(m_filters), nb_row = nb_row, nb_col = nb_col, weights=[real_filters, np.zeros(len(m_filters))])) # RELU
		imag_model.add(conv_layers.Convolution2D(activation='relu', trainable=train_gabor, input_shape=input_shape, nb_filter = len(m_filters), nb_row = nb_row, nb_col = nb_col, weights=[imag_filters, np.zeros(len(m_filters))])) # RELU
		#real_model.add(conv_layers.Convolution2D(activation='linear', trainable=train_gabor, input_shape=input_shape, nb_filter = len(m_filters), nb_row = nb_row, nb_col = nb_col, weights=[real_filters, np.zeros(len(m_filters))])) # linear
		#imag_model.add(conv_layers.Convolution2D(activation='linear', trainable=train_gabor, input_shape=input_shape, nb_filter = len(m_filters), nb_row = nb_row, nb_col = nb_col, weights=[imag_filters, np.zeros(len(m_filters))])) # linear

		# create merged model
		gabor_merged_model = models.Sequential()

		if mode == "abs":
			# merge real and imaginary models, sum up their outputs and compute squareroot
			gabor_merged_model.add(custom_layers.ExtendedMerge([real_model,imag_model], concat_axis=1, mode='abs'))
			#gabor_merged_model.add(core_layers.Merge([real_model,imag_model], concat_axis=1, output_shape=real_model.output_shape, mode=lambda x: K.sqrt(x[0]**2 + x[1]**2)))

		elif mode == "atan2":
			# merge real and imaginary models by taking their atan2
			gabor_merged_model.add(custom_layers.ExtendedMerge([real_model,imag_model], concat_axis=1, mode='atan2'))
			#gabor_merged_model.add(core_layers.Merge([real_model,imag_model], concat_axis=1, output_shape=real_model.output_shape, mode=lambda x:T.tensor.arctan2(x[1],x[0])))
			#print(gabor_merged_model.output_shape)
			#exit()

		elif mode == "abs_atan2":
			# create merged models
			abs_merge_real_imag_model = models.Sequential()
			atan2_merge_real_imag_model = models.Sequential()

			# merge real and imaginary models, sum up their outputs and compute squareroot
			abs_merge_real_imag_model.add(custom_layers.ExtendedMerge([real_model,imag_model], concat_axis=1, mode='abs'))
			atan2_merge_real_imag_model.add(custom_layers.ExtendedMerge([real_model,imag_model], concat_axis=1, mode='atan2'))
			#abs_merge_real_imag_model.add(core_layers.Merge([real_model,imag_model], concat_axis=1, output_shape=real_model.output_shape, mode=lambda x: K.sqrt(x[0]**2 + x[1]**2)))
			#atan2_merge_real_imag_model.add(core_layers.Merge([real_model,imag_model], concat_axis=1, output_shape=real_model.output_shape, mode=lambda x:T.tensor.arctan2(x[1],x[0])))

			# merge 'abs' and 'atan2' model
			gabor_merged_model.add(core_layers.Merge([abs_merge_real_imag_model, atan2_merge_real_imag_model], concat_axis=1, mode='concat'))

		else:
			raise NotImplementedError

		# append the model to the model list
		first_layer_models.append(gabor_merged_model)

	# find maximal output dimensions
	max_output_x = 0
	max_output_y = 0
	for m in first_layer_models:
		if m.output_shape[2] > max_output_x:
			max_output_x = m.output_shape[2]
		if m.output_shape[3] > max_output_y:
			max_output_y = m.output_shape[3]

	# add zero padding layers where the output dimension is smaller than the maximal output dimension
	for m in first_layer_models:
		if m.output_shape[2] < max_output_x or m.output_shape[3] < max_output_y:
			x_pad = (max_output_x - m.output_shape[2]) / 2
			y_pad = (max_output_y - m.output_shape[3]) / 2
			m.add(conv_layers.ZeroPadding2D(padding=(x_pad,y_pad)))

	# merge single models
	merged_model = models.Sequential()
	merged_model.add(core_layers.Merge(first_layer_models, concat_axis=1, mode='concat'))

	# add dropout layer
	#merged_model.add(core_layers.Dropout(0.3)) # dropout
	#merged_model.add(core_layers.Dropout(0.5)) # dropout05

	# add additional convolutional layers
	#merged_model.add(conv_layers.Convolution2D(activation="relu", init="glorot_normal", nb_filter=32, nb_col=3, nb_row=3)) # "normal" conv
	#merged_model.add(conv_layers.Convolution2D(activation="relu", init="glorot_normal", nb_filter=32, nb_col=7, nb_row=7)) # larger conv
	#merged_model.add(conv_layers.Convolution2D(activation="relu", init="glorot_normal", nb_filter=32, nb_col=15, nb_row=15)) # 2cl
	#merged_model.add(conv_layers.Convolution2D(activation="relu", init="glorot_normal", nb_filter=24, nb_col=9, nb_row=9)) # 2cl
	if add_conv2:	
		merged_model.add(conv_layers.MaxPooling2D(pool_size=(2, 2)))
		merged_model.add(conv_layers.Convolution2D(activation="relu", init="glorot_normal", nb_filter=32, nb_col=3, nb_row=3))

	# flatten model
	merged_model.add(core_layers.Flatten())

	# add two fully connected layers
	#merged_model.add(core_layers.Dense(activation="sigmoid", init="glorot_normal", output_dim=300)) # at least one convolutional layer
	merged_model.add(core_layers.Dense(activation="sigmoid", init="glorot_normal", output_dim=250)) # noconv
	merged_model.add(core_layers.Dense(activation="sigmoid", init="glorot_normal", output_dim=200))

	# add output layer
	merged_model.add(core_layers.Dense(activation="linear", init="glorot_normal", output_dim=30))
	
	# create optimizer and compile model
	optimizer = optimizers.SGD(lr=learningrate, momentum=momentum, decay=decay, nesterov=True)
	merged_model.compile(loss='mse', optimizer=optimizer)

	# return model and optimizer
	return merged_model, optimizer
