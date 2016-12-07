import numpy as numpy

class PCA(object):
	def __init__(self, data):
		self.data = data
		self.data -= self.data.mean(axis=0)

	def analyse():
		C = np.cov(self.data.T)
		self.evals, self.evecs = np.linalg.eig(C)
		indices = np.argsort(self.evals)
		indices = indices[::-1]
		self.evals = self.evals[indices]
		self.evecs = self.evecs[:, indices]

	def reduce_dim(new_dim):
		w = self.evecs[:, :new_dim]
		return np.dot(self.data, new_dim)

	def original_dim(data):
		w = self.evecs[:, :data.shape[1]]
		return np.dot(data, w.T)
