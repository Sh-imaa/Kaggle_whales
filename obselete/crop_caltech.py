# crop caltech images
import struct
import re

import cv2

def extract_data(annotation_file):
	re_label = re.compile('lbl=\'(.*)\'\s')
	re_person = re.compile('pos =\[(.*)\]\s')
	re_pos = re.compile('((\d+\.\d+.){4})')
	re_start = re.compile('str=([0-9]+)\s')
	re_end = re.compile('end=([0-9]+)\s')
	re_hide = re.compile('hide=([0-9]+)')

	person_found, pos_found = None, None
	for i, line in enumerate(open(annotation_file)):
		if re.search(re_label, line):
			label = re.search(re_label, line).group(1)
			if label != 'person':
				continue
			person_found = True
			start = re.search(re_start, line).group(1)
			end = re.search(re_end, line).group(1)
			hide = re.search(re_hide, line).group(1)
		if re.search(re_person, line):
			pos_found = True
			person = re.search(re_person, line).group(1)
			pos = re.findall(re_pos ,person)
			pos = [p[0][:-1].split() for p in pos]
		if person_found and pos_found:
			person_found, pos_found = None, None
			yield (label, start, end, hide, pos)

def main():
	images_folder = '/home/retailyze/DataSets/calTechDataSet/images/set00/V001/'
	annotation_file = '/home/retailyze/DataSets/calTechDataSet/annotations/set00/V001.txt'

	for label, start, end, hide, pos in extract_data(annotation_file):
		frame_numbers = [frame for frame in range(int(start), int(end))]
		for i, frame_number in enumerate(frame_numbers):
			img_path = images_folder + ('img%s.jpg'%frame_number)
			img = cv2.imread(img_path)
			cv2.imshow('image', img)
			cv2.waitKey(0)
			position = pos[i]
			x, y, w, h = int(float(position[0])), int(float(position[1])), int(float(position[2])), int(float(position[3]))
			ROI = img[y:y+h, x:x+w]
			cv2.imshow('ROI', ROI)
			cv2.waitKey(0)		

		raw_input('Press Enter')

if __name__ == '__main__':
	main()
