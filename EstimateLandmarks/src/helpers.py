import math

def mult_list(l, f):
	'''This function iterates of a nested list and multiplies each element by factor f'''	
	if isinstance(l, list):
		for i in l:
			mult_list(i, f)
	else:
		l *= f

def loss_to_pixel(loss, scalefactor):
	'''This method computes the distance in pixels implied by the error value'''
	return math.sqrt(loss) * scalefactor
