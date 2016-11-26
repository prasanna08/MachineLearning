import numpy as np

class NeuralNet(object):
	def __init__(self, inputs, outputs, hidden_layers, eta, target_error):
		self.inputs = np.array(inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0],1)
		self.outputs = np.array(outputs)
		if self.outputs.ndim == 1:
			self.outputs = self.outputs.reshape(self.outputs.shape[0],1)
		self.test_cases = self.inputs.shape[0]
		self.input_neurons = np.array([len(self.inputs[0])])
		self.output_neurons = np.array([len(self.outputs[0])])
		self.hidden_layers = np.array(hidden_layers)
		self.eta = eta
		self.target_error = target_error

		self.neural_layers = np.concatenate(
			[self.input_neurons, self.hidden_layers, self.output_neurons])
		self.num_layers = self.neural_layers.shape[0]
		self.theta = [np.random.rand(self.neural_layers[i+1], self.neural_layers[i]+1)
			for i in np.arange(self.num_layers-1)]

		self.activation = [np.random.rand(self.test_cases, self.neural_layers[i])
			for i in range(self.num_layers)]
		self.sig_activation = [np.random.rand(self.test_cases, self.neural_layers[i])
			for i in range(self.num_layers)]
		self.delta = [np.random.rand(self.test_cases, self.neural_layers[i])
			for i in range(self.num_layers)]
		self.activation[0] = self.inputs

	def sigmoid(self, z):
		return (1/(1+np.exp(-z)))

	def grad(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def calculate_error(self):
		y = self.outputs
		f = self.sig_activation[-1]
		return -(y*np.log(f) + (1-y)*np.log(1-f))

	def shuffle(self):
		data = np.concatenate([self.inputs, self.outputs], axis=1)
		np.random.shuffle(data)
		self.inputs = data[:, :self.inputs.shape[1]]
		self.outputs = data[:, self.inputs.shape[1]:]

	def forward_prop(self):
		# Forward propagation.
		for i in range(self.num_layers - 1):
			layer_input = np.concatenate(
				[np.ones((self.test_cases,1)), self.sig_activation[i]], axis=1)
			self.activation[i+1] = np.dot(layer_input, self.theta[i].T)
			self.sig_activation[i+1] = self.sigmoid(self.activation[i+1])

	def back_prop(self):
		self.delta[self.num_layers-1] = (self.sig_activation[-1] - self.outputs)
		for i in range(self.num_layers-1, 1, -1):
			self.delta[i-1] = (
				np.dot(self.delta[i], self.theta[i-1])[:, 1:]) * self.grad(self.activation[i-1])

	def update_theta(self):
		for i in range(self.num_layers-1, 0, -1):
			diff = np.dot(
				self.delta[i].T, np.concatenate(
					[np.ones((self.test_cases,1)), self.sig_activation[i-1]], axis=1))
			diff = (self.eta * diff) / self.test_cases
			self.theta[i-1] -= diff

	def train(self):
		epoch = 1
		error = np.inf
		while error >= self.target_error:
			#self.shuffle()
			self.forward_prop()
			error = self.calculate_error()
			error = np.sum(error)/self.test_cases
			self.back_prop()
			self.update_theta()
			if ((epoch+1) % 100) == 0:
				print "Error (%d epoch) = %f" % (epoch+1, error)
			epoch += 1