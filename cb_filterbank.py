import numpy as np
import scipy.io as sio

from os import listdir
from os.path import join

from ICF_all import *

import logging

def generate_cb_bank(size=(4,4)):
	"""Generates combination of checkboarder pattern for the size given,
	this follow the paper "Filtered Channel Features for Pedestrian Detection".
	Parameters:
		size: tuple(int, int), the size of maximum checkboarder pattern.

	Returns: list of array, all bank combinations."""
	max_h, max_w = size

	bank = []
	for h in range(1, max_h+1):
		for w in range(1, max_w+1):
			# just for square filters
			if h == w:
				f = np.ones((h,w))
				bank.append(f)

			# just for non vector/column filters
			if h != 1 and w != 1:
				f = np.ones((h,w))
				f[::2, ::2] = -1
				f[1::2, 1::2] = -1
				bank.append(f)

			for j in range(1, h):
				f = np.ones((h,w))
				f[:j] = -1
				bank.append(f)

			for i in range(1, w):
				f = np.ones((h,w))
				f[:, :i] = -1
				bank.append(f)

	return bank

def generate_cb_features(model_size, cell_size, bank):
	"""Generates all convloution of bank filters to the model size.
	Parameters:
		model_size: tuple(int, int), height and width of the model.
		cell_size: tuple(int, int), the cell size, where the cell is
				   the unit the model will divide into.
		bank: list of array, all the filters to be used.

	Returns: list of array, features to be used."""
	h = int(model_size[0]/cell_size[0])
	w = int(model_size[1]/cell_size[1])

	features = []
	for f in bank:
		f_h, f_w = f.shape
		for j in range(h-f_h+1):
			for i in range(w-f_w+1):
				features.append((i, j, f))

	return features

def integrate_channel(channel, cell_size):
	"""Integrates a channel in a way that each cell is integrated into one pixel/value.
	Parameters:
		channel: array, channel to integrate.
		cell_size: tuple(int, int), the height and width of the cell to integrate based ubon.

	Returns: array, integrated channel of size equals the size of channel divided by area of cell.
		"""
	max_h, max_w = channel.shape[0]/cell_size[0], channel.shape[1]/cell_size[1]
	reshaped = channel[:max_h*cell_size[0], :max_w*cell_size[1]].reshape(cell_size[0], cell_size[1], max_h, max_w)

	i_channel = np.sum(reshaped, (0,1))
	return i_channel

def evaluate_cb_features(i_channels, features):
	features_cb = []
	for channel in i_channels:
		for i, j, f in features:
			j_m, i_m = f.shape
			channel_subset = channel[j:j_m+j, i:i_m+i]		
			feature = np.sum(channel_subset*f, (0,1))
			features_cb.append(feature)

	return features_cb

def evaluate_all_cb_features(source_folder, saving_path, max_feature_size=(4,4),
							 model_size=(160,96), cell_size=(6,6)):
	bank = generate_cb_bank(max_feature_size)
	features = generate_cb_features(model_size, cell_size, bank)
	del bank
	feature_mat = []
	for i, image_name in enumerate(listdir(source_folder)):
		logging.info('Processing image_no: %s, image_name: %s' %(i, image_name[:-4]))
		channels = sio.loadmat(join(source_folder, image_name))["channels"]
		features_cb = evaluate_cb_features(channels, features)
		del channels
		feature_mat.append(features_cb)
		del features_cb
	feature_mat = np.array(feature_mat)
	np.save(saving_path, feature_mat)