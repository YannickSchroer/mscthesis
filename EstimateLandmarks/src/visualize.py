# -*- coding: utf-8 -*-

import os
import numpy as np
import math
import PIL.Image as pil
import PIL.ImageDraw as pildraw

def drawCross(im, pos, size, color):
	draw = pildraw.Draw(im)
	
	# line 1
	topleft = (pos[0] - size, pos[1] - size)
	bottomright = (pos[0] + size, pos[1] + size)

	# line 2
	bottomleft = (pos[0] - size, pos[1] + size)
	topright= (pos[0] + size, pos[1] - size)

	# draw lines
	draw.line((topleft, bottomright), fill=color)
	draw.line((bottomleft, topright), fill=color)

	del draw

def drawLine(im, pos0, pos1, color):
	draw = pildraw.Draw(im)

	draw.line((pos0, pos1), fill=color)

	del draw

def visualize_predictions(image_list, predictions, true_labels=None, crosssize=2, predsave = None, predshow = False, color_pred = (255,255,255), color_true = (0,0,0)):
	'''This method visualizes the predictions. It takes a path and saves to it all images with marks at the predicted positions. If the pred path equals None, the images are not saved but displayed.'''
	if len(image_list) != predictions.shape[0] or predictions.shape[1] % 2 != 0:
		raise Exception #TODO specify exceptions

	for idx, image in enumerate(image_list):
		im = pil.open(image['path'])

		n = int(predictions.shape[1] / 2)
		
		for i in range(n):
			drawCross(im, (predictions[idx][2*i], predictions[idx][2*i + 1]), crosssize, color_pred)

		if true_labels != None:
			color_line = ((color_pred[0] + color_true[0]) / 2, (color_pred[1] + color_true[1]) / 2, (color_pred[2] + color_true[2]) / 2)
			for i in range(n):
				drawCross(im, (true_labels[idx][2*i], true_labels[idx][2*i + 1]), crosssize, color_true)
				drawLine(im, (true_labels[idx][2*i], true_labels[idx][2*i + 1]), (predictions[idx][2*i], predictions[idx][2*i + 1]), color_line)

		if predshow:
			im.show()
		if predsave != None and os.path.isdir(predsave):
			if (predsave[-1] != "/"):
				predsave += "/"
			im.save(predsave + image['path'][image['path'].rfind('/'):-3] + "png")

def loss_histogram(losses, path, nb_bins=20):
	'''This method creates a histogram of the loss distribution. The path argument does not take an extension.'''	
	print("Visualize losses as histogram, saving histogram to \"" + path + ".png\"")
	# calculate minimum and maximum
	l_min = np.min(losses)
	l_max = np.max(losses)
	margin = (l_max - l_min) / (10000 * nb_bins)
	l_max += margin
	
	# initialize bins
	bins = np.zeros(nb_bins, dtype=int)

	# assign each loss to a bin
	for l in losses:
		b = int(math.floor( ((l - l_min) / (l_max - l_min)) * nb_bins ))
		bins[b] += 1

	# calculate bin width
	width = (l_max - l_min) / nb_bins

	# create data file for the plot
	data_file_str = ""
	for idx, i in enumerate(bins):
		data_file_str += str(round(l_min + (idx * width), 4)) + "\t" + str(i) + "\n"
	data_file_str = data_file_str[:-1]

	# save data file
	with open(path + ".txt", "w") as data_file:
		data_file.write(data_file_str)

	# create gnuplot file
	gnu_file_str = "set term png size 3840,2160 truecolor\n"
	gnu_file_str += "set output '" + path + ".png'\n"
	gnu_file_str += "set xlabel 'Loss'\n"
	gnu_file_str += "set ylabel 'Number of appearances'\n"
	gnu_file_str += "set grid\n"
	gnu_file_str += "set boxwidth 0.9 relative\n"
	gnu_file_str += "set xrange [" + str(l_min) + ":" + str(l_max) +"]\n"
	gnu_file_str += "set xtics font ', 10'\n"
	gnu_file_str += "set xtics " + str(l_min) + "," + str(width) + "\n"
	gnu_file_str += "set style fill transparent solid 0.5 noborder\n"
	gnu_file_str += "plot '" + path + ".txt' using ($1+" + str(width) + "/2.0):2 with boxes lc rgb'green' notitle"

	# save gnuplot file
	with open(path + ".plt", "w") as gnu_file:
		gnu_file.write(gnu_file_str)

	# run gnuplot script
	os.system("gnuplot " + path + ".plt")

	# remove gnuplot file
	os.remove(path + ".plt")
