from keras import backend as K

def scaled_tanh(x):
	return K.tanh(x) * 640
