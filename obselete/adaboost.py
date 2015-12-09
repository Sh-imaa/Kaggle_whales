# adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
from os.path import join

from sklearn.externals import joblib

def main():
	body_model = False
	ICF = True

	if body_model:
		saving_loc = '/home/retailyze/Downloads/INRIAPerson/body_model/svm/'
		train_loc = join(saving_loc, 'train.npy')
		val_loc = join(saving_loc, 'val.npy')
		training_set = np.load(train_loc)
		val_set = np.load(val_loc)
	elif ICF:
		saving_loc = '/home/retailyze/Downloads/INRIAPerson/svm/adaboost/'
		train_loc = join(saving_loc, 'train.npy')
		val_loc = join(saving_loc, 'val.npy')
		training_set = np.load('/home/retailyze/Downloads/INRIAPerson/NormalizedWithPadding/svm/train.npy')
		print training_set.shape
		val_set = np.load('/home/retailyze/Downloads/INRIAPerson/NormalizedWithPadding/svm//val.npy')
		print val_set.shape

	print "starting training"
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=100)
	adaboost_saving_loc = join(saving_loc,'adaboost_not_cropped_100.pkl')
	bdt.fit(training_set[:, 1:], training_set[:, 0])
	joblib.dump(bdt, adaboost_saving_loc)

	print "starting predicting"
	predicted = bdt.predict(val_set[:, 1:])
	print confusion_matrix(predicted, val_set[:, 0])

if __name__ == "__main__":
	main()