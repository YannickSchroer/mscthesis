import sys
import os

#os.system("python src/train.py layouts/etl_eyes_96_128.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -s weights/etl_eyes_96_128/01_zappa_rmsprop 1> outputs/etl_eyes_96_128/01_output.txt 2> outputs/etl_eyes_96_128/01_errors.txt")
for i in range(10,21):
	num = str(i).zfill(2)
	numLast = str(i - 1).zfill(2)
	os.system("python src/train.py layouts/etl_eyes_96_128.l data/MUCT_fixed/muct-landmarks/MUCT_TRAIN.csv -l weights/etl_eyes_96_128/" + numLast + "_zappa_rmsprop -s weights/etl_eyes_96_128/" + num + "_zappa_rmsprop 1> outputs/etl_eyes_96_128/" + num + "_output.txt 2> outputs/etl_eyes_96_128/" + num + "_errors.txt")
