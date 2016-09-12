import os
import time
import pickle
import numpy as np

import nn
import dataset_io
import helpers
import visualize

data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TEST_KAGGLE_REDUCED.csv"
weight_load_path = "weights/gabor_lr0.1_sqrt_16_09_08_zappa"
gabor_file = "data/gabor/gabor_filters.dat"
learningrate = 0
decay = 0
batchsize = 4
normalize = 2
normalize_output = True
resolution = (96,128)
grayscale = False

# load gabor filters
try:
	with open(gabor_file, 'rb') as gabor_file:
		gabor_filters = pickle.load(gabor_file)
except IOError:
	print('Error while loading gabor filters')

# build model
model, optimizer = nn.build_gabor_model(gabor_filters, learningrate = learningrate, decay = decay, mode="atan2", add_conv2=False)

# Load status
dataset_io.load_status(model, optimizer, weight_load_path + "/1000")

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
expanded_x_test = [x_test,x_test,x_test,x_test,x_test,x_test,x_test,x_test,x_test,x_test]

# normalize output
if normalize_output:
	y_test /= max_dim

#model.fit(expanded_x_test, y_test, nb_epoch=1, batch_size=batchsize, shuffle=True, verbose=True)

# test model
score = model.evaluate(expanded_x_test, y_test, 1, verbose=True)
print('Test score:', score)
print('Test score in pixels:', helpers.loss_to_pixel(score, np.max(resolution)))

# predict
predictions = model.predict(expanded_x_test, batchsize)

if normalize_output:
	predictions *= max_dim
	y_test *= max_dim

visualize.visualize_predictions(image_list, predictions, y_test, crosssize=5, predsave = "visualizations/gabor_lr0.1_sqrt_16_09_08_zappa/test", predshow=False, color_pred = (255,255,255), color_true=(0,0,0))
