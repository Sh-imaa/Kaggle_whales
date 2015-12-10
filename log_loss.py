# Multi class log-loss
import numpy as np
from math import log

def multi_class_log_loss(true, pred):
	pred = np.maximum(10**-5, np.minimum(1-10**-5, pred))
	N = true.shape[0]
	loss = - (1/float(N)) * np.sum(np.sum((true*np.log(pred)),1),0)
	return loss

def main():
	true = np.array([[1,0,0],[0,1,0],[0,1,0]])
	pred1 = np.array([[0.9,0.1,0],[0.4,0.6,0],[0.2,0.5,0.3]])
	pred2 = np.array([[0.9,0.1,0],[0.4,0.6,0],[0,1,0]])

	print 'with pred1'
	print multi_class_log_loss(true, pred1)

	print 'with pred2'
	print multi_class_log_loss(true, pred2)

if __name__ == '__main__':
	main()