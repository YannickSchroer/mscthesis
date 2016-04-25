import sys
import os

for i in range(16,18):
	num = str(i).zfill(2)
	numLast = str(i - 1).zfill(2)
	os.system("python src/train.py layouts/etl_eyes_72_96.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl_eyes_72_96/" + numLast + "_zappa_rmsprop -s weights/etl_eyes_72_96/" + num + "_zappa_rmsprop 1> outputs/etl_eyes_72_96/" + num + "_output.txt 2> outputs/etl_eyes_72_96/" + num + "_errors.txt")
