# -------SART algorithm---------
# implementation: Python
# author: Ren
# version: 0
# last modified: 18/07/2019
# ------------------------------

import cv2 as cv
import numpy as np
from sart import dataset, sart
from construction import constructor
from calibrator import artagTool

dataset_root = './calibration/rabbit/raw/'
dataset_dir = dataset_root+'refined/'
datafile_prefix = 'myProject_'
datafile_postfix = '.jpeg'
datafile_indexformat = '{:0>3d}'
datafile_format = dataset_dir + datafile_prefix + datafile_indexformat + datafile_postfix
volume_a = 128

def show_image(image):
	cv.namedWindow('show_image')
	cv.imshow("image", image)
	cv.waitKey(0)

if __name__ == '__main__':
	# artagTool = artagTool(
	# 		calibration_folder='calibration/rabbit/photo/',
 #            standard_path='./calibration/rabbit/photo/c00.ppm',
 #            sample_path='./calibration/rabbit/sample/c00.ppm',
 #            raw_path=dataset_root,
 #            dataset_roi=[],
	# 	)
	# artagTool.run()
	# dat = dataset(200, datafile_format)
	# algo = sart(dat,volume_a)
	constructor = constructor('./log/log.json')
	constructor.draw_error_curve('./log/errorlog.json')
	
	