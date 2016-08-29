import pickle
import numpy as np
import math
import cmath
import PIL.Image
import time

def get_k(k_max, alpha, L, m, l):
	'''This method computes the vector k for given k_max, alpha, L, l and m.'''
	return ( k_max * math.pow(alpha,-m) * math.cos(math.pi * l / L) , k_max * math.pow(alpha,-m) * math.sin(math.pi * l / L))

def create_gabor(k, sigma):
	'''This method computes a gabor wavelet for a given vector k.'''
	gabor_filter = np.empty((101,101), dtype=complex)

	k_squared = k[0] * k[0] + k[1] * k[1]
	sigma_squared = sigma * sigma
	k_div_sigma = k_squared / sigma_squared

	for i in range(-50,51):
		for j in range(-50,51):
			x_squared = i * i + j * j
			gabor_filter[i+50,j+50] = k_div_sigma * math.exp(-(k_squared * x_squared) / (2 * sigma_squared)) * (cmath.exp(1j * (k[0] * i + k[1] * j)) - math.exp(- sigma_squared / 2))

	return gabor_filter

def visualize_filter(gabor_filter):
	'''This method opens a window with an image created from a (3*x*y) color image.'''
	image = np.copy(gabor_filter)
	image -= np.min(image)
	image *= 255 / np.max(image)
	foo = PIL.Image.fromarray(image)
	foo.show()

def cut_filter(gabor_filter, left_idx = None, top_idx = None):
	'''This method crops a filter so that the border with close to 0 values are dismissed'''
	cut_filter = np.copy(gabor_filter)
	limit = 0.001
	if left_idx == None and top_idx == None:
		left_idx, top_idx= 0, 0
		for i in range(cut_filter.shape[0]):
			if np.max(cut_filter[i,:]) < limit:
				top_idx = i
			else:
				break
		for j in range(cut_filter.shape[1]):
			if np.max(cut_filter[:,j]) < limit:
				left_idx = j
			else:
				break
	cut_filter = cut_filter[top_idx:cut_filter.shape[0]-top_idx,left_idx:cut_filter.shape[1]-left_idx]
	return cut_filter

# define values for relevant variables
k_max = math.pi / 2
alpha = math.sqrt(2)
sigma = 2 * math.pi
L = 8
M = 5
cut_indices_top = cut_indices_left = [37,35,31,27,23]

gabor_filters = []
#real_filters = []
#imag_filters = []

# create filters and store them to a list
for m in range(M):
	gabor_filters_row = []
	for l in range(L):
		k = get_k(k_max, alpha, L, m, l)
		gabor_filter = create_gabor(k, sigma)
		gabor_filters_row.append(cut_filter(gabor_filter, cut_indices_left[m], cut_indices_top[m]))
		#real = np.real(gabor_filter)
		#imag = np.imag(gabor_filter)
		#real_filters.append(cut_filter(real, cut_indices_left[m], cut_indices_top[m]))
		#imag_filters.append(cut_filter(imag, cut_indices_left[m], cut_indices_top[m]))
	gabor_filters.append(gabor_filters_row)

# save filters to the file system
with open("data/gabor/gabor_filters.dat", 'wb') as gabor_file:
	#pickle.dump([real_filters, imag_filters], gabor_file, pickle.HIGHEST_PROTOCOL)
	pickle.dump(gabor_filters, gabor_file, pickle.HIGHEST_PROTOCOL)
