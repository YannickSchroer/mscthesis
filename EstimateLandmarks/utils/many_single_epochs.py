import sys
import os

os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -s weights/etl/01_zappa_rmsprop 1> outputs/etl/01_output.txt 2> outputs/etl/01_errors.txt")
for i in range(2,10):
	os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl/0" + str(i - 1) + "_zappa_rmsprop -s weights/etl/" + str(i) + "_zappa_rmsprop 1> outputs/etl/0" + str(i) + "_output.txt 2> outputs/etl/0" + str(i) + "_errors.txt")
