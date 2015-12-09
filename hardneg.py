#  hard negatives
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
from Scene_pred import *

import logging

def hard_neg(neg_folder, features, model, mean, std):

	hard_negs = {}
	for i, image_name in enumerate(listdir(neg_folder)):
		logging.info('Processing image_no: %s, image_name: %s' %(i, image_name[:-4]))
		img_path = join(neg_folder, image_name)
		img = cv2.imread(img_path)
		possible_humans = predict_img(img, model.decision_function, 0, features, cv2.integral, evaluate_all_features,
								  	  mean=mean, std=std, cell_size=(1,1), scale=2, downscale=True,
								  	  model_size=(160,60), step_size_h=10, step_size_w=10, feature_type='acf')
		hard_negs[img_path] = possible_humans
		logging.info('Number of false pos: %s' %(len(possible_humans)))
	return hard_negs


def main():
	neg_folder = '/home/retailyze/Downloads/INRIAPerson/Train/neg/'
	features = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/features_rand.npy')
	model = joblib.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/svm_rbf_scaled.pkl')
	mean = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/mean.npy')
	std = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/std.npy')
	hard_negs = hard_neg(neg_folder, features, model, mean, std)
	sio.savemat('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/hardNeg', hard_negs)

if __name__ == '__main__':
	main()