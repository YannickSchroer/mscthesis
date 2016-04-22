import sys
import os

for i in range(3,10):
	os.system("python src/train.py layouts/estimate_landmarks_tutorial.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl/_0" + str(i - 1) + "_zappa_rmsprop -s weights/etl/" + str(i) + "_zappa_rmsprop 1> outputs/etl/" + str(i-1) + "_output.txt 2> outputs/etl/" + str(i-1) + "_errors.txt")
