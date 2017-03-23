import KMeans
import numpy as np
import preprocess

class RBF(model.SupervisedModel):
	def __init__(self, n_inputs, n_hidden, n_outputs, sigma):
		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs
		self.sigma = sigma
		self.weights = preprocess.xavier_init((n_hidden, n_outputs))

	def get_clusters_distances(self, data):
		self.km = KMeans.KMeans(data, clusters=self.n_hidden)
		self.km.train()
		clusters, distance = self.km.assign_clusters(data)
		return distance

	def train(self, data, labels):
		distance = self.get_clusters_distances(data)
		hidden = np.exp(-(distance**2)/(2*self.sigma**2))
		self.weights = np.dot(np.linalg.pinv(hidden), labels)

	def get_outputs(self, inputs):
		_, distance = self.km.assign_clusters(inputs)
		hidden = np.exp(-(distance**2)/(2*self.sigma**2))
		return np.dot(hidden, self.weights)
