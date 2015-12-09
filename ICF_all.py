import cv2
import numpy as np
import scipy.io as sio
 
from math import pi
from os.path import join
from os import listdir

import random
import itertools

import logging


def generate_icf_features(num_features, channel_count=10, model_size=(160, 60), min_w=5, min_h=5):
	"""Generates random icf features within the bondary of the model size.
	Parameters:
		num_features: int, number of features to be generated.
		channel_count: int, number of channels that will be used.
		model_size: tuple(int, int), the size of the model, this will be tthe bondary
					within which the rectangler features to be generated.
		min_w: int, the minimum width of the generated rectangler feature.
		min_h: int, the minimum height of the generated rectangler feature.

	Returns: List of (int, int, int, int, int) <-> (channel_index, x1, y1, x2, y2)
			 where the tuple  consists of the channel to apply the feature on,
			 the x coordinate of the top left of the feature,
			 the y coordinate of the top left of the feature,
			 the x coordinate of the bottom right of the feature,
			 the y coordinate of the bottom right  of the feature respectively."""
	features = []
	for i in range(num_features):
		channel_index = random.randint(0,channel_count-1)
		x1 = random.randint(0, model_size[1] - min_w -1)
		y1 = random.randint(0, model_size[0] - min_h -1)
		x2 = random.randint(x1 + min_w, model_size[1]-1)
		y2 = random.randint(y1 + min_h, model_size[0]-1)
		features.append((channel_index, x1, y1, x2, y2))

	return features

def generate_acf_features(channel_count=10, model_size=(160, 60), scale=4):
	"""Generates acf features within the bondary of the model size.
	Parameters:
		channel_count: int, number of channels that will be used.
		model_size: tuple(int, int), the size of the model, this will be tthe bondary
					within which the rectangler features to be generated.
		scale: int, the scale to scale down to.

	Returns: List of (int, int, int, int) <-> (c,  x, y, s)
			 where the tuple  consists of the channel to apply the feature on,
			 the x coordinate of the top left of the feature,
			 the y coordinate of the top left of the feature,
			 the scale it was scaled down to.

	Notes: - Default params will generate 10*(160/4)*(60/4) = 6000 feature.
		   - The function can be reduced to just scaling down instead of cretaing
		   	 features then using integrated images, but this is used to be done 
		   	 in the exact way as ICF"""
	x = range(0, model_size[1] - scale, scale)
	y = range(0, model_size[0] - scale, scale)
	c = range(0, channel_count)
	s = [scale]
	return list(itertools.product(c, x, y, s))

def evaluate_feature(integral_channels, feature_type, feature):
	"""Evlauates a feature given the feature, its type, and the image channels.

	Parameters:
		integral_channels: List of array, list of channels of the input image.
						   The number of channels is expected to agree with
						   the number of channels used when generating the feature.
		feature_type: String, either "icf", "acf".
		feature: tuple, the tuple structure depends on the feature type.

	Returns: int, the feature.
	"""
	if feature_type == "icf":
		ch, x1, y1, x2, y2 = feature
		# a - b - c + d
		integral_channel = integral_channels[ch]
		a = integral_channel[y1, x1]
		b = integral_channel[y1, x2]
		c = integral_channel[y2, x1]
		d = integral_channel[y2, x2]

		return a - b - c + d

	elif feature_type == "acf":
		ch, x1, y1, scale = feature
		integral_channel = integral_channels[ch]
		a = integral_channel[y1, x1]
		b = integral_channel[y1+scale, x1]
		c = integral_channel[y1, x1+scale]
		d = integral_channel[y1+scale, x1+scale]

		return (a - b - c + d)/scale**2

def evaluate_all_features(integral_channels, feature_list, feature_type):
	"""Evaluate al features in the list given.
	Parameters:
		integral_channels: List of array, list of channels of the input image.
						   The number of channels is expected to agree with
						   the number of channels used when generating the feature.
		feature_type: String, either "icf" or "acf".
		feature: List of tuples, the tuple structure depends on the feature type.

	Returns: List of int, list of features.
	"""
	features = []
	for feature in feature_list:
		features.append(evaluate_feature(integral_channels, feature_type, feature))
	return features


def evaluate_folder(source_directory, saving_path, feature_list, feature_type='icf'):
	feature_mat = []
	for i, image_name in enumerate(listdir(source_directory)):
		logging.info('Processing image_no: %s, image_name: %s' %(i, image_name[:-4]))
		channels = sio.loadmat(join(source_directory, image_name))["channels"]
		features = evaluate_all_features(channels, feature_list, feature_type)
		del channels
		feature_mat.append(features)
		del features
	feature_mat = np.array(feature_mat)
	np.save(saving_path, feature_mat)

def generateIntegralChannels(img, integration_fn=cv2.integral, save=False, saving_loc='', colored=True, **kwargs):
	"""Generates 10(or 7 if grey image) integrated channels for image and save them if needed.
	Parameters:
		img: array, the image to compute the channels for.
		integration_fn: function, function used to inetgrate the channels.
		save: bool, flag to save the channels.
		saving_loc: string, where to save the channels if "save" is set.
		colored: bool, if the image is colored.
		**kwargs: keyword arguments to be passed to integration_fn.

	Returns: array, array of channels, only if "save" isn't set."""
	if save:
		if saving_loc == '':
			raise ValueError("saving_loc must not be empty if save is True")
	channels = [] # list of array
	if colored:
		if len(img.shape) < 2:
			raise ValueError("The image must be colored if colored is True")
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
		for i in range(3):
			channels.append(integration_fn(img_luv[:,:,i], **kwargs))
	else:
		img = img_gray

	x_der = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
	y_der = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
	mag, angle = cv2.cartToPolar(x_der, y_der)

	no_rows = mag.shape[0]
	no_cols = mag.shape[1]

	hist = []
	for i in range(6):
		hist.append(np.zeros((no_rows, no_cols)))

	for row in range(no_rows):
		for col in range(no_cols):
			ang = angle[row, col] * 180 / pi
			if ang >= 180:
				ang -= 180
			elif ang < 0:
				ang += 180

			ind = int(ang/30)

			hist[ind][row, col] = mag[row, col]

	for i in range(len(hist)):
		hist[i] = integration_fn(hist[i], **kwargs)
	channels += hist
	channels.append(integration_fn(mag, **kwargs))

	if save:
		sio.savemat(saving_loc, mdict={'channels': np.array(channels)})
	else: 
		return np.array(channels)

def generate_all_channels(source_directory, saving_directory, integration_fn=cv2.integral, **kwargs):
	"""Generates all channels for a given directory and saves them.
	Parameters:
		source_directory: string, the images' directory.
		saving_directory: string, the location to save channels at.
		integration_fn: function, function used to inetgrate the channels.
		**kwargs: keyword parameters to be passed to integration_fn."""
	for img_file in  listdir(source_directory):
		if img_file[-4:] != ".jpg" and img_file[-4:] != ".png":
			continue
		img = cv2.imread(join(source_directory, img_file))
		img_name = img_file[:-4] + '.mat'
		saving_loc = join(saving_directory, img_name)
		generateIntegralChannels(img, integration_fn, True, saving_loc, **kwargs)

def acf_features_by_resizing(channels, scale_acf=4):
	features = []
	for channel in channels:
		new_size = channel.shape[1]/scale_acf, channel.shape[0]/scale_acf
		features.append(cv2.resize(channel, new_size, interpolation=cv2.INTER_AREA).flatten())

	return [feature for channel_f in features for feature in channel_f]

def acf_resize_file(source_directory, saving_path, scale_acf=4):
	feature_mat = []
	for i, image_name in enumerate(listdir(source_directory)):
		logging.info('Processing image_no: %s, image_name: %s' %(i, image_name[:-4]))
		channels = sio.loadmat(join(source_directory, image_name))["channels"]
		features = acf_features_by_resizing(channels, scale_acf)
		del channels
		feature_mat.append(features)
		del features
	feature_mat = np.array(feature_mat)
	np.save(saving_path, feature_mat)
