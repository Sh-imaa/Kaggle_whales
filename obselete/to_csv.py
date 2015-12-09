# generate csv val and train
import scipy.io as sio
import numpy as np
from os.path import join

def main():
	pos_file = sio.loadmat("/home/retailyze/Downloads/INRIAPerson/Train/cropped/gaussianR1pos/icf.mat")["icf"]
	neg_file =sio.loadmat("/home/retailyze/Downloads/INRIAPerson/extraNeg/cropped/gaussianR1neg/icf.mat")["icf"]

	pos_icf =  np.concatenate((np.ones((pos_file.shape[0],1)), pos_file), 1) 
	neg_icf =  np.concatenate((np.zeros((neg_file.shape[0],1)), neg_file), 1)
	data_icf = np.concatenate((pos_icf, neg_icf), 0)
	np.random.shuffle(data_icf)

	training_size = int(0.8*data_icf.shape[0])
	training_set = data_icf[:training_size, :]
	val_set = data_icf[training_size:, :]

	features_path = '/home/retailyze/Downloads/INRIAPerson/features/'
	np.save(join(features_path, 'train_extra_neg.npy'), training_set)
	np.save(join(features_path, 'val_extra_neg.npy'), val_set)
	np.savetxt(join(features_path, 'train_extra_neg.csv'), training_set, delimiter=",")
	np.savetxt(join(features_path, 'val_extra_neg.csv'), val_set, delimiter=",")

if __name__ == '__main__':
	main()