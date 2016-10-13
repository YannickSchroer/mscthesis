import os
import time
import pickle
import numpy as np

import nn
import dataset_io
import custom_callbacks

result_string = "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"

data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TRAIN_KAGGLE_REDUCED.csv"
folder_name = "gabor_lr0.1_atan2_hd_gray_2cl_complete"
weight_store_path = "weights/" + folder_name
gabor_file = "data/gabor/gabor_filters.dat"
learningrate = 0.1
decay = 0.
batchsize = 4
epochs = 500
normalize = 2
normalize_output = True
resolution = (120,160)
grayscale = True
mode = "atan2"
add_conv2 = False

# load gabor filters
try:
	with open(gabor_file, 'rb') as gabor_file:
		gabor_filters = pickle.load(gabor_file)
except IOError:
	print('Error while loading gabor filters')

# build model
model, optimizer = nn.build_gabor_model(gabor_filters, input_shape=(1 if grayscale else 3, resolution[0], resolution[1]), learningrate = learningrate, decay = decay, mode=mode, add_conv2 = add_conv2)

# Load status
#dataset_io.load_status(model, optimizer, weight_store_path + "/1000") # LOAD =========================================================================
dataset_io.load_status(model, optimizer, "weights/gabor_lr0.1_atan2_hd_gray_2cl/1000") # LOAD BEST GABOR NETWORK

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
expanded_x_train = [x_train for i in range(10)]

# normalize output
if normalize_output:
	y_train /= max_dim

# deep copy data and labels
x_train_orig = np.copy(x_train)
y_train_orig = np.copy(y_train)

# add callbacks
if normalize_output:
	callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train_orig, y_train_orig, x_train.shape[0], (float(original_resolution[0]) / float(max_dim), float(original_resolution[1]) / float(max_dim)))]
else:
	callbacks = [custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], original_resolution)]
loss_callback = custom_callbacks.RecordLossGabor(epochs, nb_labels, x_train_orig, y_train_orig, resolution, model)
callbacks.append(loss_callback)

# fit model
model.fit(expanded_x_train, y_train, callbacks=callbacks, nb_epoch=epochs, batch_size=batchsize, shuffle=True, verbose=True)

# save weights
dataset_io.store_status(model, optimizer, weight_store_path + "/1500")

result_string += "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"
result_string += "epochs: " + str(epochs) + "\n"
result_string += "grayscale: " + str(grayscale) + "\n"
result_string += "weightscales: "
result_string += "glorot_normal\n"
result_string += "learningrate: " + str(learningrate) + "\n"
result_string += "decay: " + str(decay)

result_string += "\n\n"

# save loss history from callbacks
result_string += "Loss histories:\n"
for l in loss_callback.loss_history:
	result_string += str(l) + ","
result_string = result_string[:-1] + "\n"

with open("results/" + folder_name + "/results_1500.dat", "w") as loss_file:
		loss_file.write(result_string)
