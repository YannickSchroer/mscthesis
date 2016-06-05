import os
import numpy as np

import nn
import dataset_io
import helpers

weightscales = [0.001, 0.01, 0.1, 1, 10]
learningrates = [0.002, 0.01, 0.1, 0.2]
decays = [0, 0.001]

data_path = "data/MUCT_fixed/muct-landmarks/MUCT_TRAIN_XS.csv"
layout_path = "layouts/etl_eyes_96_128.l"
batchsize = 4
epochs = 25

print('Loading model from {0}'.format(layout_path))
layout = nn.load_layout(layout_path)

# Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}'.format(data_path, resolution[0], resolution[1]))
x_train, y_train = dataset_io.read_data(data_path, resolution, None, normalize=True)

# normalize labels
if False:
	y_train[:,0] /= 480
	y_train[:,1] /= 640
	y_train[:,2] /= 480
	y_train[:,3] /= 640

# do grid search
for w in weightscales:
	for l in learningrates:
		for d in decays:
			print("Learning rate: ", l)
			print("Weight scale: ", w)
			print("Decay: ", d)
			os.mkdir("weights/gridsearch_500/coarse/w" + str(w) + "_l" + str(l) + "_d" + str(d)) #TODO Remove comment
		
			# create model and optimizer
			model, optimizer = nn.build_model_to_layout(layout, l, decay=d)

			# scale weights
			weights = model.get_weights()
			helpers.mult_list(weights, w * 20)
			model.set_weights(weights)

			#predictions = model.predict(x_train, batch_size = batchsize, verbose=False)
			#print(np.round(predictions, 3))
			#raw_input("Press Enter to continue...")
			#continue

			# Train the model
			print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], batchsize, epochs))
			for i in range(epochs):
				print("Epoch: " +  str(i + 1))
				model.fit(x_train, y_train, nb_epoch=1, batch_size=batchsize, shuffle=False, verbose=True)
				loss = model.evaluate(x_train, y_train, 1, verbose=False)
				dataset_io.store_status(model, optimizer, "weights/gridsearch_500/coarse/w" + str(w) + "_l" + str(l) + "_d" + str(d) + "/w" + str(w) + "_l" + str(l) + "_d" + str(d) + "_e" + str(i + 1) + "_s" + str(loss))
