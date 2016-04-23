import sys
import os

#os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -s weights/etl/01_zappa_rmsprop 1> outputs/etl/01_output.txt 2> outputs/etl/01_errors.txt")
for i in range(10,21):
	z = str(i - 1)
	if i == 10:
		z = "09"
	os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl/" + z + "_zappa_rmsprop -s weights/etl/" + str(i) + "_zappa_rmsprop 1> outputs/etl/" + str(i) + "_output.txt 2> outputs/etl/" + str(i) + "_errors.txt")
