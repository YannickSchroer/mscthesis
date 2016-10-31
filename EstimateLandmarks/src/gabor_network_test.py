import os
import time
import pickle
import numpy as np

import nn
import dataset_io
import helpers
import visualize

training  = False	# True: calculate training error, False: calculate test error
draw_pred = False	# True: create images with predictions False: do not create images with predictions

gabor_file = "data/gabor/gabor_filters.dat"
data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TRAIN_KAGGLE_REDUCED.csv" if training else "data/MUCT_fixed/muct-landmarks/MUCT_TEST_KAGGLE_REDUCED.csv"
folder_name = "gabor_lr0.1_absatan2_hd_gray_noconv"
weight_load_path = "weights/" + folder_name
load_epoch = 1000
learningrate = 0
decay = 0
batchsize = 4
normalize = 2
normalize_output = True
resolution = (96,128)
#resolution = (120,160)
grayscale = True
mode = "abs_atan2"
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
dataset_io.load_status(model, optimizer, weight_load_path + "/" + str(load_epoch))

# Print from where the images are loaded, to which resolution they are scaled and whether they are normalized
if normalize == 1:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [0,1]'.format(data_path, resolution[0], resolution[1]))
elif normalize == 2:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [-1,1]'.format(data_path, resolution[0], resolution[1]))
else:
	print('Loading data from {0} and rescaling it to {1}x{2}. Images are not normalized!'.format(data_path, resolution[0], resolution[1]))

# Load data
x_test, y_test, image_list, original_resolution = dataset_io.read_data(data_path, resolution, normalize=normalize, grayscale=grayscale, return_original_resolution=True, return_image_properties=True)
nb_labels = 30
max_dim = np.max(original_resolution)
expanded_x_test = [x_test for i in range(10)]

# normalize output
if normalize_output:
	y_test /= max_dim

#model.fit(expanded_x_test, y_test, nb_epoch=1, batch_size=batchsize, shuffle=True, verbose=True)

# test model
score = model.evaluate(expanded_x_test, y_test, 1, verbose=True)
print('Test score:', score)
print('Test score in pixels:', helpers.loss_to_pixel(score, np.max(resolution)))

if draw_pred:
	# predict
	print('Predict landmarks')
	predictions = model.predict(expanded_x_test, batchsize)

	if normalize_output:
		predictions *= max_dim
		y_test *= max_dim

	visualizations_path = "visualizations/" + folder_name
	visualizations_train_path = visualizations_path + "/train_" + str(load_epoch)
	visualizations_test_path = visualizations_path + "/test_" + str(load_epoch)

	if not os.path.isdir(visualizations_path):
		os.makedirs(visualizations_path)
	if training:
		if not os.path.isdir(visualizations_train_path):
			os.makedirs(visualizations_train_path)
	else:
		if not os.path.isdir(visualizations_test_path):
			os.makedirs(visualizations_test_path)

	print('Visualize landmarks')
	visualize.visualize_predictions(image_list, predictions, y_test, crosssize=5, predsave = visualizations_train_path if training else visualizations_test_path, predshow=False)
