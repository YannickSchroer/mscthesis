# this function iterates of a nested list and multiplies each element by factor f
def mult_list(l, f):
	if isinstance(l, list):
		for i in l:
			mult_list(i, f)
	else:
		l *= f
