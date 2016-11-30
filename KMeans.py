import numpy as np

class KMeans(object):
	def __init__(self, inputs, clusters):
		self.inputs = np.array(inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0], 1)

		self.test_cases = self.inputs.shape[0]
		self.num_clusters = clusters
		self.clusters = self.inputs[np.random.choice(self.test_cases, clusters, replace=False), :]
		self.cluster_dist = np.array([np.zeros((1, self.test_cases)) for i in range(self.num_clusters)])

	def calculate_cluster_distances(self):
		for i in range(self.num_clusters):
			self.cluster_dist[i] = np.sqrt(np.sum((self.inputs - self.clusters[i])**2, axis=1))

	def assign_data_points(self):
		new_clusters = np.zeros(self.clusters.shape)
		point_counts = np.zeros((self.num_clusters, 1))
		for i in range(self.test_cases):
			b_cluster = np.where(self.cluster_dist[:, :, i] == self.cluster_dist[:,:,i].min())[0][0]
			new_clusters[b_cluster] += self.inputs[i]
			point_counts[b_cluster] += 1

		self.clusters[point_counts>0] = new_clusters[point_counts>0] / point_counts[point_counts>0]

	def train(self):
		while True:
			old_clusters = self.clusters.copy()
			self.calculate_cluster_distances()
			self.assign_data_points()
			if np.sum(old_clusters - self.clusters) == 0: break
