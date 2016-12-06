import numpy as np
from scipy import linalg as LA

class LDA(object):
	def __init__(self, data_inputs, data_labels):
		self.data_inputs = np.array(data_inputs)
		self.data_labels = data_labels
		self.test_cases = self.data_inputs.shape[0]
		self.labels = np.unique(data_labels)
		self.Sw = np.zeros((self.data_inputs.shape[1], self.data_inputs.shape[1]))
		self.Sb = self.Sw.copy()

	def analyse(self):
		C = np.cov(self.data_inputs.T)

		for label in self.labels:
			indices = np.where(self.data_labels == label)
			points = self.data_inputs[indices[0]]
			classcov = np.cov(points.T)
			self.Sw += (np.float(points.shape[0])/self.test_cases) * classcov

		self.Sb = C - self.Sw
		evals, evecs = LA.eig(self.Sw, self.Sb)
		indices = np.argsort(evals)
		indices = indices[::-1]
		evals = evals[indices]
		evecs = evecs[indices]
		self.eigen_vals = evals
		self.eigen_vecs = evecs

	def reduce_dim(self, red_n, data_inputs=None):
		w = self.eigen_vecs[:,:red_n]
		if data_inputs is None:
			data_inputs = self.data_inputs
		return np.dot(data_inputs, w)

	def expand_dim(self, red_data):
		red_n = red_data.shape[1]
		return np.transpose(np.dot(self.eigen_vecs[:,:red_n], red_data.T))
