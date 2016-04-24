import sys
import os

for i in range(2,6):
	z = str(i - 1)
	os.system("python src/train.py layouts/etl_eyes_72_96.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl_eyes_72_96/0" + z + "_zappa_rmsprop -s weights/etl_eyes_72_96/0" + str(i) + "_zappa_rmsprop 1> outputs/etl_eyes_72_96/0" + str(i) + "_output.txt 2> outputs/etl_eyes_72_96/0" + str(i) + "_errors.txt")
