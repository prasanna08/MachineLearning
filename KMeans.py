import numpy as np

class KMeans(object):
	def __init__(self, inputs, clusters):
		self.inputs = np.array(inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0], 1)
		self.shuffle()
		self.test_cases = self.inputs.shape[0]
		self.num_clusters = clusters
		self.clusters = self.inputs[np.random.choice(self.test_cases, clusters, replace=False), :]
		self.cluster_dist = np.zeros((self.test_cases, self.num_clusters))

	def shuffle(self):
		np.random.shuffle(self.inputs)

	def calculate_cluster_distances(self):
		for i in range(self.num_clusters):
			self.cluster_dist[:, i] = np.sqrt(np.sum((self.inputs - self.clusters[i])**2, axis=1))

	def assign_data_points(self):
		assigned_cluster = np.where(self.cluster_dist.argsort(axis=1) == 0)[1]
		for i in range(self.num_clusters):
			points = self.inputs[assigned_cluster == i]
			self.clusters[i] = points.sum(axis=0)/points.shape[0]

	def train(self):
		while True:
			old_clusters_norm = np.linalg.norm(self.clusters, axis=1)
			self.shuffle()
			self.calculate_cluster_distances()
			self.assign_data_points()
			new_clusters_norm = np.linalg.norm(self.clusters, axis=1)

			if np.isnan(self.clusters).any() == True:
				self.clusters = self.inputs[np.random.choice(
					self.test_cases, self.num_clusters, replace=False), :]
			elif (old_clusters_norm == new_clusters_norm).all() == True:
				break

	def assign_clusters(self, inputs):
		cluster_dist = np.zeros((inputs.shape[0], self.num_clusters))
		for i in range(self.num_clusters):
			cluster_dist[:, i] = np.sqrt(np.sum((inputs - self.clusters[i])**2, axis=1))
		assigned_clusters = np.where(cluster_dist.argsort(axis=1) == 0)[1]
		return assigned_clusters, cluster_dist
