import numpy as np

class SOM(object):

	def __init__(
			self, data, x, y, eta_b=0.3, eta_n=0.1, n_radius=0.5,
			eta_b_final=0.03, eta_n_final=0.01, n_radius_final=0.05):
		"""A 2D self organising map.
		"""
		self.data = data
		self.n_dim = data.shape[1]
		self.x = x
		self.y = y
		self.eta_b = eta_b
		self.eta_n = eta_n
		self.n_radius = n_radius
		self.eta_b_final = eta_b_final
		self.eta_n_final = eta_n_final
		self.n_radius_final = n_radius_final
		self.map_dim = 2

		self.f_map = np.mgrid[0:1:np.complex(0, x), 0:1:np.complex(0, y)]
		self.f_map = self.f_map.reshape(self.map_dim, x*y)

		# Update weights randomly.
		self.weights = np.random.rand(x*y, self.n_dim)
		self.map_dist = np.zeros((x*y, x*y))

		for i in range(x*y):
			self.map_dist[i, :] = np.sqrt(np.sum((self.f_map - self.f_map[:, i].reshape(self.f_map.shape[0], 1)) ** 2, axis=0))

	def calc_dist(self, inp, weights):
		return np.sqrt(np.sum((inp - weights) ** 2, axis=1))

	def train(self, epoch):
		self.eta_b_init = self.eta_b
		self.eta_n_init = self.eta_n
		self.n_radius_init = self.n_radius

		for i in range(epoch):
			for j in range(self.data.shape[0]):
				dist = self.calc_dist(self.data[j, :], self.weights)
				best = dist.argmin()
				# Update weight of best match.
				self.weights[best, :] += self.eta_b * (self.data[j, :] - self.weights[best, :])
				# Find neighbours og best match.
				neighbours = np.where(self.map_dist[best, :] <= self.n_radius,1,0)
				neighbours[best] = 0
				neighbours = neighbours.reshape(neighbours.shape[0], 1)
				# Update weights of neighbour nodes.
				self.weights += self.eta_n * neighbours * (self.data[j, :] - self.weights)


			# Decrease neighbour learning rate and best match learning rate.
			self.eta_b = self.eta_b_init * np.power(self.eta_b_final / self.eta_b_init, float(i)/epoch)
			self.eta_n = self.eta_n_init * np.power(self.eta_n_final / self.eta_n_init, float(i)/epoch)
			# Decrease neighbour hood radius.
			self.n_radius = self.n_radius_init * np.power(self.n_radius_final / self.n_radius_init, float(i)/epoch)

	def predict(self, inp):
		dist = self.calc_dist(inp, self.weights)
		best = dist.argmin()
		return best

	def predict_all(self, data):
		best = np.zeros((data.shape[0], 1))
		for i in range(data.shape[0]):
			best[i] = self.predict(data[i, :])
		return best
