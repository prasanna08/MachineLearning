import numpy as np

class Kmeansnet(object):
	def __init__(self, data, clusters, eta):
		self.data = data
		self.n_dim = data.shape[1]
		self.num_clusters = clusters
		self.weights = np.random.rand(self.num_clusters, self.n_dim)
		self.eta = eta

	def calc_dist(self, inp, weights):
		return np.sum((weights * inp), axis=1)

	def normalise_data(self, data):
		normalisers = np.sqrt(np.sum(data ** 2, axis=1)).reshape(self.data.shape[0], 1)
		return data / normalisers

	def train(self, epochs):
		self.data = self.normalise_data(self.data)

		for i in range(epochs):
			for d in range(self.data.shape[0]):
				dist = self.calc_dist(self.data[d, :], self.weights)
				cluster = np.argmax(dist)
				self.weights[cluster, :] += self.eta * self.data[d, :] - self.weights[cluster, :]

	def predict(self, inp):
		dist = self.calc_dist(inp, self.weights)
		best = np.argmax(dist)
		return best

	def predict_all(self, data):
		best = np.zeros((data.shape[0], 1))
		for i in range(data.shape[0]):
			best[i] = self.predict(data[i, :])
		return best
