import os
import time
import pickle
import numpy as np

import nn
import dataset_io
import custom_callbacks

#for momentum in [0.1,0.5,0.9]:
for learningrate in [0.2]:
	# initialize result string
	result_string = "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"

	# define variables
	# learningrate = 0.2
	momentum = 0

	data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TRAIN_KAGGLE_REDUCED.csv"
	folder_name = "cnn/cnn_2maxpooling_" + str(learningrate)
	epochs = 400
	load_epoch = 0
	save_epoch = load_epoch + epochs
	weight_store_path = "weights/" + folder_name
	result_store_path = "results/" + folder_name
	batchsize = 4
	decay = 0.
	normalize = 2
	normalize_output = True
	resolution = (120,160)
	grayscale = False
	initialization = "glorot_normal"
	activation_function = "sigmoid"
	nb_max_pooling = 2

	# create folders for weights and results if they do not exist
	if not os.path.isdir(weight_store_path):
		os.makedirs(weight_store_path)
	if not os.path.isdir(result_store_path):
		os.makedirs(result_store_path)

	# build model
	model, optimizer = nn.build_cnn_model(input_shape=(1 if grayscale else 3, resolution[0], resolution[1]), learningrate = learningrate, momentum=momentum, decay = decay, initialization=initialization, activation_function=activation_function, nb_max_pooling=nb_max_pooling)

	# Load status
	if load_epoch > 0:
		dataset_io.load_status(model, optimizer, weight_store_path + "/" + str(load_epoch)) # LOAD =========================================================================

	# Print from where the images are loaded, to which resolution they are scaled and whether they are normalized
	if normalize == 1:
		print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [0,1]'.format(data_path, resolution[0], resolution[1]))
	elif normalize == 2:
		print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [-1,1]'.format(data_path, resolution[0], resolution[1]))
	else:
		print('Loading data from {0} and rescaling it to {1}x{2}. Images are not normalized!'.format(data_path, resolution[0], resolution[1]))

	# Load data
	x_train, y_train, original_resolution = dataset_io.read_data(data_path, resolution, normalize=normalize, grayscale=grayscale, return_original_resolution=True)
	nb_labels = 30
	max_dim = np.max(original_resolution)
	#expanded_x_train = [x_train for i in range(10)]

	# normalize output
	if normalize_output:
		y_train /= max_dim

	# deep copy data and labels
	x_train_orig = np.copy(x_train)
	y_train_orig = np.copy(y_train)

	# add callbacks
	if normalize_output:
		normalized_resolution = (float(original_resolution[0]) / float(max_dim), float(original_resolution[1]) / float(max_dim))
		callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train_orig, y_train_orig, x_train.shape[0], normalized_resolution)]
	else:
		callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], original_resolution)]
	loss_callback = custom_callbacks.RecordLossCNN(epochs, nb_labels, x_train_orig, y_train_orig, resolution, model)
	callbacks.append(loss_callback)

	# fit model
	model.fit(x_train, y_train, callbacks=callbacks, nb_epoch=epochs, batch_size=batchsize, shuffle=True, verbose=True)

	# save weights
	dataset_io.store_status(model, optimizer, weight_store_path + "/" + str(save_epoch))

	# construct result string
	result_string += "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"
	result_string += "epochs: " + str(epochs) + "\n"
	result_string += "grayscale: " + str(grayscale) + "\n"
	result_string += "initialization: " + initialization + "\n"
	result_string += "activation function: " + activation_function + "\n"
	result_string += "batch size: " + str(batchsize) + "\n"
	result_string += "max pooling: " + str(max_pooling) + "\n"
	result_string += "learningrate: " + str(learningrate) + "\n"
	result_string += "decay: " + str(decay) + "\n"
	result_string += "resolution: " + str(resolution[0]) + "x" + str(resolution[1])

	# save loss history from callbacks
	loss_string = ""
	for l in loss_callback.loss_history:
		loss_string += str(l) + "\n"
	loss_string = loss_string[:-1]

	# save results data file and results csv file
	with open(result_store_path + "/results_" + str(save_epoch) + ".dat", "w") as result_file:
		result_file.write(result_string)
	with open(result_store_path + "/results_" + str(save_epoch) + ".csv", "w") as loss_file:
		loss_file.write(loss_string)
