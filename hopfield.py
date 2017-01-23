import numpy as np

class Hopfield(object):
	def __init__(self, data, b=1.0, synchronous=False, random=True):
		data = 2 * data - 1

		# Hebb learning.
		self.weights = np.dot(data.T, data)
		self.weights = self.weights / data.shape[0]
		np.fill_diagonal(self.weights, 0)
		
		self.synchronous = synchronous
		self.random = random
		b = self.b

	def set_inputs(self, inps):
		inps = 2 * inps - 1
		self.activation = inps

	def update_neuron(self, activation):
		if self.synchronous:
			activation = np.tanh(self.b * np.sum(self.weights * activation, axis=1))
			activation = np.where(activation > 0, 1, -1)
		else:
			order = range(self.activation.shape[0])
			if self.random:
				np.random.shuffle(order)
			for i in order:
				activation[i] = np.tanh(self.b * np.sum(self.weights[i] * activation))
		return activation

	def get_output(self, inps, max_iter):
		self.set_inputs(inps)
		for i in range(max_iter):
			self.activation = self.update_neuron(self.activation)
		return (self.activation + 1) / 2

	def compute_energy(self):
		return -1 * np.dot(self.activation, np.dot(self.weights, self.activation))
