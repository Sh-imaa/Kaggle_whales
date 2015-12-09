# Image Test
import numpy as np
from sklearn import svm
from skimage.transform import pyramid_expand, pyramid_reduce
from sklearn.externals import joblib
import scipy.io as sio

from os import listdir
from os.path import join

from ICF_all import *
from cb_filterbank import *
from nms import *

import logging

def predict_img(img, prediction_fn, prediction_thresh, features, integration_fn,
				feature_eval_fn, mean=None, std=None, cell_size=(1,1), scale=1, downscale=True,
				model_size=(160,60), step_size_h=20, step_size_w=20, type='', **kwargs):
	# this preprocessing step is done in training, so it's recommended to do it here.
	# can be replaced/removed based on the training method

	possible_humans = []
	img = cv2.GaussianBlur(img, (1,1), 0)
	if downscale:
		new_size = int(img.shape[1]/scale), int(img.shape[0]/scale)
	else:
		new_size = int(img.shape[1]*scale), int(img.shape[0]*scale)
	img = cv2.resize(img, new_size)
	if type == 'cd':
		channels = generateIntegralChannels(img, integration_fn=integration_fn, save=False,
										    saving_loc='', colored=True, cell_size=cell_size)
	else:
		channels = generateIntegralChannels(img, integration_fn=integration_fn, save=False,
										    saving_loc='', colored=True)
	channels = np.asarray(channels)

	h, w = img.shape[:2]
	window_size = model_size[0]/cell_size[0], model_size[1]/cell_size[1]
	ch_h, ch_w = channels.shape[:2]
	for x in range(0, w-window_size[1], step_size_w):
		for y in range(0, h-window_size[0], step_size_h):
			try:
				test_existance = channels[0][y+window_size[0], x+window_size[1]]
			except:
				continue
			window_channels = channels[:, y:y+window_size[0], x:x+window_size[1]]
			features_values = feature_eval_fn(window_channels, features, **kwargs)
			features_values = np.asarray(features_values)
			features_values = (features_values - mean) / std
			prediction = prediction_fn(features_values)
			orig_x, orig_y = x*cell_size[0], y*cell_size[1]
			top_left_corner, right_bottom_corner = (orig_x,orig_y), (orig_x+model_size[1], orig_y+model_size[0])
			if prediction > prediction_thresh:
				print "found human candidate", prediction
				possible_humans.append((orig_x, orig_y, orig_x+model_size[1], orig_y+model_size[0], prediction))
				cv2.rectangle(img, top_left_corner, right_bottom_corner, (0,255,0))
				center = (orig_x+model_size[1]/2, orig_y+model_size[1]/2)
				cv2.circle(img, center, 3, (225,0,100))
			else:
				print prediction
				cv2.rectangle(img, top_left_corner, right_bottom_corner, (255,0,0))
			cv2.imshow("Image", img)
			cv2.waitKey(50)
	if downscale: 
		return np.asarray(possible_humans) * scale
	else:
		return np.asarray(possible_humans) / scale

def main():
	img = cv2.imread('/home/retailyze/features/CF/article-0-1B9A78C5000005DC-950_308x425.jpg')

	features = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/features_rand.npy')
	model = joblib.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/svm_rbf_scaled.pkl')
	mean = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/mean.npy')
	std = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/std.npy')
	possible_humans = predict_img(img, model.decision_function, 0, features, cv2.integral, evaluate_all_features,
								  mean=mean, std=std, cell_size=(1,1), scale=1, downscale=True,
								  model_size=(160,60), step_size_h=20, step_size_w=20, feature_type='acf')
	possible_humans = np.asarray(possible_humans)

	picked = non_max_suppression_fast(possible_humans, 0.05, by_pred=False, max_bd=True)
	# new_size = int(img.shape[1]/2), int(img.shape[0]/2)
	# img = cv2.resize(img, new_size)
	for box in picked:
		cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),  (0,255,0))
		center = (box[0]+((box[2]-box[0])/2), box[1]+((box[3]-box[1])/2))
		cv2.circle(img, center, 3, (0,0,0))
		cv2.circle(img, center, 5, (2550,255,255))
	cv2.imshow("nms",img)
	cv2.waitKey(0)

if __name__ == '__main__':
	main()