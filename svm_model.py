# svm
from sklearn import svm
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.externals import joblib

from os.path import join
import logging

logging.basicConfig(level=logging.INFO)
def main():
	pos_features_path = '/home/retailyze/Downloads/INRIAPerson/checkb/cropped/svm/featuresPos160_60.npy'
	neg_features_path = '/home/retailyze/Downloads/INRIAPerson/checkb/cropped/svm/featuresNeg160_60.npy'

	saving_loc = '/home/retailyze/Downloads/INRIAPerson/checkb/cropped/svm/'

	pos_features = np.load(pos_features_path)[:, 0::3]
	neg_features = np.load(neg_features_path) [:, 0::3]
	train, val = prepare_features(pos_features, neg_features, True, saving_loc)
	del pos_features
	del neg_features

	clf = svm.SVC(kernel='rbf')

	logging.info('starts training')
	clf.fit(train[:, 1:], train[:, 0])
	del train
	logging.info('starts predicting')
	predicted = clf.predict(val[:, 1:])
	conf_mat = confusion_matrix(predicted, val[:, 0])
	acc = accuracy_score(val[:, 0], predicted)
	del val
	del predicted
	logging.info('Confusion matrix: %s' %conf_mat)
	logging.info('Accuracy: %s' %acc)
	logging.info('saving model')
	joblib.dump(clf, join(saving_loc, 'svm_rbf_scaled.pkl'))


def prepare_features(pos_features, neg_features, save=False, saving_loc=''):
	pos = np.concatenate((np.ones((pos_features.shape[0],1)), pos_features), 1)
	neg = np.concatenate((np.zeros((neg_features.shape[0],1)), neg_features), 1)

	data = np.concatenate((pos, neg), 0)
	del pos
	del neg
	mean = np.mean(data[:, 1:], 0)
	std = np.std(data[:, 1:], 0)
	data = np.concatenate((np.reshape(data[:, 0], (data.shape[0],1)), preprocessing.scale(data[:, 1:])), 1)
	np.random.shuffle(data)

	# test samples are in seprate folder
	train_size = data.shape[0] * 0.8
	train = data[:train_size, :]
	val = data[train_size:, :]
	del data

	if save:
		if saving_loc == '':
			raise ValueError("saving_loc must not be empty if save is True")
		else:
			train_loc = join(saving_loc, 'train.npy')
			val_loc = join(saving_loc, 'val.npy')
			mean_loc = join(saving_loc, 'mean.npy')
			std_loc = join(saving_loc, 'std.npy')
			np.save(train_loc, train)
			np.save(val_loc, val)
			np.save(mean_loc, mean)
			np.save(std_loc, std)
	return train, val

if __name__ == "__main__":
	main()