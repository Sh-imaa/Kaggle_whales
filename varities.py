from os.path import join
from os import listdir

import cv2
import numpy as np

def change_size(source_directory, output_directory, new_size):
	for img_name in listdir(source_directory):
		img_ext = img_name[-4:]
		if img_ext != '.png' and img_ext != '.jpg':
			continue
		image = cv2.imread(join(source_directory, img_name))
		new_image = cv2.resize(image, new_size)
		cv2.imwrite(join(output_directory, img_name), new_image)

def apply_gaussian(source_directory, output_directory, kernel=(1,1)):
	for img_name in listdir(source_directory):
		img_ext = img_name[-4:]
		if img_ext != '.png' and img_ext != '.jpg':
			continue
		image = cv2.imread(join(source_directory, img_name))
		new_image = cv2.GaussianBlur(image, kernel, 0)
		cv2.imwrite(join(output_directory, img_name), new_image)

def remove_padding(source_directory, output_directory, (x1, y1), (x2, y2)):
	for img_name in listdir(source_directory):
		img_ext = img_name[-4:]
		if img_ext != '.png' and img_ext != '.jpg':
			continue
		image = cv2.imread(join(source_directory, img_name))
		image = image[y1:y2, x1:x2]
		cv2.imwrite(join(output_directory, img_name), image)