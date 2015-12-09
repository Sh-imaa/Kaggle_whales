# Test functions
from ICF_all import *
from cb_filterbank import *
from varities import *
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
# pos_path = '/home/retailyze/Downloads/INRIAPerson/ACF/90_45/pos_imgs/'
# neg_path = '/home/retailyze/Downloads/INRIAPerson/ACF/90_45/neg_imgs/'

# saving_path_pos = '/home/retailyze/Downloads/INRIAPerson/ACF/90_45/pos/'
# saving_path_neg = '/home/retailyze/Downloads/INRIAPerson/ACF/90_45/neg/'

# generate_all_channels(neg_path, saving_path_neg)


# source_folder = '/home/retailyze/Downloads/INRIAPerson/ACF/120_60/pos/'
# saving_path = '/home/retailyze/Downloads/INRIAPerson/ACF/120_60/svm/featuresPos120_60.npy'
# evaluate_all_cb_features(source_folder, saving_path, model_size=(120,60))


# features = generate_acf_features(channel_count=10, model_size=(160,60), scale=4)
# features = generate_icf_features(50000, channel_count=10, model_size=(160, 60), min_w=5, min_h=5)

# np.save('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/features_rand.npy', np.asarray(features))

# features = np.load('/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/features_rand.npy')

# source_folder = '/home/retailyze/Downloads/INRIAPerson/ICF/cropped/neg/'
# saving_path = '/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm/featuresNeg160_60.npy'
# evaluate_folder(source_folder, saving_path, features, feature_type='acf')

# source_folder = '/home/retailyze/Downloads/INRIAPerson/ICF/cropped/pos/'
# saving_path = '/home/retailyze/Downloads/INRIAPerson/ACF/cropped/svm2/featuresPos160_60.npy'
# acf_resize_file(source_folder, saving_path)

# source_directory = '/home/retailyze/Downloads/INRIAPerson/Train/cropped/neg/'
# output_directory = '/home/retailyze/Downloads/INRIAPerson/ACF/90_45/neg_imgs/'
# change_size(source_directory, output_directory, (45,90))

