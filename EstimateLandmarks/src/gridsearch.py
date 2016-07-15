import os
import time
import numpy as np

import nn
import dataset_io
import helpers
import custom_callbacks
import visualize

loss_string = "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"

weightscales = ["glorot_normal"]
learningrates = [0.05,0.1]
momentums = [0,0.001,0.1]

loss_matrix = np.zeros((len(momentums), len(learningrates)))
test_loss_matrix = np.zeros((len(momentums), len(learningrates)))

data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TRAIN_KAGGLE_REDUCED.csv"
test_data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TEST_KAGGLE_REDUCED.csv"
layout_path = "layouts/etl_kaggle_96_128_tutorial_glorot_normal.l"
batchsize = 4
epochs = 150
normalize = 2
normalize_output = True
grayscale = False

print('Loading model from {0}'.format(layout_path))
layout = nn.load_layout(layout_path)

# Get input shape and resolution
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]

# Print from where the images are loaded, to which resolution they are scaled and whether they are normalized
if normalize == 1:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [0,1]'.format(data_path, resolution[0], resolution[1]))
elif normalize == 2:
	print('Loading data from {0} and rescaling it to {1}x{2}. Input images are normalized to [-1,1]'.format(data_path, resolution[0], resolution[1]))
else:
	print('Loading data from {0} and rescaling it to {1}x{2}. Images are not normalized!'.format(data_path, resolution[0], resolution[1]))

# Load data
x_train, y_train, original_resolution = dataset_io.read_data(data_path, resolution, normalize=normalize, grayscale=grayscale, return_original_resolution=True) # change
nb_labels = layout[-1][1]['output_dim']
max_dim = np.max(original_resolution)
x_test, y_test = dataset_io.read_data(test_data_path, resolution, normalize=normalize, grayscale=grayscale) # change

if normalize_output:
	y_train /= max_dim
	y_test /= max_dim

loss_callbacks = []

# do grid search
for m_idx, m in enumerate(momentums):
#for w_idx, w in enumerate(weightscales):
	for l_idx, l in enumerate(learningrates):	
		print("Learning rate: ", l)
		#print("weightscale: ", w)
		print("momentum: ", m)
		model, optimizer = nn.build_model_to_layout(layout, learningrate=l, momentum=m, decay=0.0) #change

		if normalize_output:
			callbacks = [custom_callbacks.StopEarly(3), custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], (float(original_resolution[0]) / float(max_dim), float(original_resolution[1]) / float(max_dim)))]
		else:
			callbacks = [custom_callbacks.StopEarly(3), custom_callbacks.Distortions(x_train, y_train, x_train.shape[0], original_resolution)]
		loss_callback = custom_callbacks.RecordLoss(epochs, nb_labels, x_train, y_train, resolution, model, grayscale)
		callbacks.append(loss_callback)
		loss_callbacks.append(loss_callback)

		# scale weights
		#weights = model.get_weights()
		#helpers.mult_list(weights, w * 20)
		#model.set_weights(weights)

		# Train the model
		print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], batchsize, epochs))
		model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batchsize, callbacks=callbacks, shuffle=True, verbose=True)

		loss = model.evaluate(x_train, y_train, batch_size=batchsize, verbose=False)
		test_loss = model.evaluate(x_test, y_test, batch_size=batchsize, verbose=False)
		print(loss)
		print(test_loss)
		loss_matrix[m_idx, l_idx] = loss #change
		test_loss_matrix[m_idx, l_idx] = test_loss # change

		predictions = model.predict(x_train, batch_size = batchsize, verbose=False)
		print(np.round(predictions - y_train, 3))
		continue

loss_string += "time: " + time.strftime("%d/%m/%Y") + " - " + time.strftime("%H:%M:%S") + "\n"
loss_string += "epochs: " + str(epochs) + "\n"
loss_string += "grayscale: " + str(grayscale) + "\n"
#loss_string += "momentum: " + str(momentums[0]) + "\n"
loss_string += "weightscales: "
loss_string += "glorot_normal\n"
#for w in weightscales:
#	loss_string += str(w) + ", "
l#oss_string = loss_string[:-2] + "\n"
loss_string += "learningrates: "
for l in learningrates:
	loss_string += str(l) + ", "
loss_string = loss_string[:-2] + "\n"
loss_string += "momentums: "
for m in momentums:
	loss_string += str(m) + ", "
loss_string = loss_string[:-2] + "\n"

# save loss matrix
loss_string += "Training error:\n"
for i in range(len(momentums)):
	if i != 0:
		loss_string += "\n"
	for j in range(len(learningrates)):
		loss_string += str(loss_matrix[i,j]) + "\t"

loss_string += "\n\n"

# save test loss matrix
loss_string += "Test error:\n"
for i in range(len(momentums)):
	if i != 0:
		loss_string += "\n"
	for j in range(len(learningrates)):
		loss_string += str(test_loss_matrix[i,j]) + "\t"

loss_string += "\n\n"

# save loss history from callbacks
loss_string += "Loss histories:\n"
for lc in loss_callbacks:
	for l in lc.loss_history:
		loss_string += str(l) + ","
	loss_string = loss_string[:-1] + "\n"

print(loss_matrix)
print(test_loss_matrix)

with open("weights/gray_vs_color__color.dat", "w") as loss_file:
		loss_file.write(loss_string)
