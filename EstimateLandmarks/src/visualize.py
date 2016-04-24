# -*- coding: utf-8 -*-

import os
import PIL.Image as pil
import PIL.ImageDraw as pildraw

def drawCross(im, pos, size):
	draw = pildraw.Draw(im)
	
	# line 1
	topleft = (pos[0] - size, pos[1] - size)
	bottomright = (pos[0] + size, pos[1] + size)

	# line 2
	bottomleft = (pos[0] - size, pos[1] + size)
	topright= (pos[0] + size, pos[1] - size)

	# draw lines
	draw.line((topleft, bottomright), fill=(255,255,255))
	draw.line((bottomleft, topright), fill=(255,255,255))

	del draw

def visualize_predictions(image_list, predictions, crosssize=2, predsave = None, predshow = False):
	'''This method visualizes the predictions. It takes a path and saves to it all images with marks at the predicted positions. If the pred path equals None, the images are not saved but displayed.'''
	if len(image_list) != predictions.shape[0] or predictions.shape[1] % 2 != 0:
		raise Exception #TODO specify exceptions

	for idx, image in enumerate(image_list):
		im = pil.open(image['path'])

		for i in range(int(predictions.shape[1] / 2)):
			drawCross(im, (predictions[idx][2*i], predictions[idx][2*i + 1]), crosssize)

		if predshow:
			im.show()

		if predsave != None and os.path.isdir(predsave):
			if (predsave[-1] != "/"):
				predsave += "/"
			im.save(predsave + image['path'][image['path'].rfind('/'):])

	
