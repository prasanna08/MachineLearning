import numpy as np
import KMeans

class RBF(object):
	def __init__(self, inputs, outputs, sigma):
		self.inputs = np.array(inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0], 1)

		self.outputs = np.array(outputs)
		if self.outputs.ndim == 1:
			self.outputs = self.outputs.reshape(self.outputs.shape[0], 1)

		self.sigma = sigma
		self.hidden = np.zeros(self.outputs.shape)
		self.theta = np.random.rand(self.outputs.shape[1], self.outputs.shape[1])

	def get_clusters_distances(self):
		self.km = KMeans.KMeans(self.inputs, clusters=self.outputs.shape[1])
		self.km.train()
		clusters, distance = self.km.assign_clusters(self.inputs)
		return distance

	def train(self):
		distance = self.get_clusters_distances()
		self.hidden = np.exp(-(distance**2)/(2*self.sigma**2))
		self.theta = np.transpose(np.dot(np.linalg.pinv(self.hidden), self.outputs))

	def get_outputs(self, inputs):
		_, distance = self.km.assign_clusters(inputs)
		hidden = np.exp(-(distance**2)/(2*self.sigma**2))
		return np.dot(hidden, self.theta.T)
