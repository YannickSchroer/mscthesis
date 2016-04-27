import sys
import os

folder = "etl_eyes_48_64_lr001"
#os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -s weights/" + folder + "/01_zappa_rmsprop 1> outputs/" + folder + "/01_output.txt 2> outputs/" + folder + "/01_errors.txt")
for i in range(10,41):
	num = str(i).zfill(2)
	numLast = str(i - 1).zfill(2)
	os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/" + folder + "/" + numLast + "_zappa_rmsprop -s weights/" + folder + "/" + num + "_zappa_rmsprop 1> outputs/" + folder + "/" + num + "_output.txt 2> outputs/" + folder + "/" + num + "_errors.txt")
