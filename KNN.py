import numpy as np 

'''Python implementation of KNN classifier 
   with L1 distance'''

class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		'''X is N * D where each row is an example. y is 1D of size N'''
		self.Xtr = X
		self.ytr = y

	def predict(self, X):
		'''X is N * D where each row is an example we wish to predict label for '''
		num_test = X.shape[0]
		#make sure output type is same as input type
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		#loop over all test rows 
		for i in xrange(num_test):
			#find nearest training image to the i-th test image - use L1 distance (Sum of absolute value diff)
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
			min_index = np.argmin(distances) # get the index with smallest distance
			Ypred[i] = self.ytr[min_index] # predict label of the nearest example 

		return Ypred
		