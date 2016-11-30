import numpy as np

class NeuralNet(object):
	BATCH_LEARNING = 'batch'
	MINI_BATCH_LEARNING = 'mini_batch'
	STOCHASTIC_LEARNING = 'stochastic'
	def __init__(
			self, train_inputs, train_outputs, hidden_layers, validation_inputs,
			validation_outputs, eta=0.7, momentum=0.3, early_stopping=True,
			method=BATCH_LEARNING, mini_batch_size=None):
		# Prepare train_data.
		self.inputs = np.array(train_inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0],1)
		self.outputs = np.array(train_outputs)
		if self.outputs.ndim == 1:
			self.outputs = self.outputs.reshape(self.outputs.shape[0],1)

		# Prepare validation data for early stopping.
		if early_stopping:
			self.validation_inputs = np.array(validation_inputs)
			if self.validation_inputs.ndim == 1:
				self.validation_inputs = self.validation_inputs.reshape(
					self.validation_inputs.shape[0],1)
			self.validation_outputs = np.array(validation_outputs)
			if self.validation_outputs.ndim == 1:
				self.validation_outputs = self.validation_outputs.reshape(
					self.validation_outputs.shape[0], 1)

		self.test_cases = self.inputs.shape[0]
		self.input_neurons = np.array([len(self.inputs[0])])
		self.output_neurons = np.array([len(self.outputs[0])])
		self.hidden_layers = np.array(hidden_layers)
		self.eta = eta
		self.momentum = momentum
		self.early_stopping = early_stopping

		self.learning_method = method
		if (self.learning_method == self.MINI_BATCH_LEARNING and
				mini_batch_size == None):
			raise Exception(
				"Please enter mini_batch_size if using "
				"MINI_BATCH_LEARNING method")
		self.mini_batch_size = mini_batch_size

		self.neural_layers = np.concatenate(
			[self.input_neurons, self.hidden_layers, self.output_neurons])
		self.num_layers = self.neural_layers.shape[0]
		self.theta = [np.random.rand(self.neural_layers[i+1], self.neural_layers[i]+1)
			for i in np.arange(self.num_layers-1)]
		self.old_updates = [np.zeros((self.neural_layers[i+1], self.neural_layers[i]+1))
			for i in np.arange(self.num_layers-1)]

		if self.learning_method == self.BATCH_LEARNING:
			self.inputs_per_batch = self.test_cases
		elif self.learning_method == self.MINI_BATCH_LEARNING:
			self.inputs_per_batch = self.mini_batch_size
		elif self.learning_method == self.STOCHASTIC_LEARNING:
			self.inputs_per_batch = 1

		self.activation = [np.random.rand(self.inputs_per_batch, self.neural_layers[i])
			for i in range(self.num_layers)]
		self.sig_activation = [np.random.rand(self.inputs_per_batch, self.neural_layers[i])
			for i in range(self.num_layers)]
		self.delta = [np.random.rand(self.inputs_per_batch, self.neural_layers[i])
			for i in range(self.num_layers)]

	def sigmoid(self, z):
		return (1/(1+np.exp(-z)))

	def grad(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def calculate_error(self, target, output):
		return np.sum(-(target*np.log(output) + (1-target)*np.log(1-output)))/target.shape[0]

	def _shuffle(self):
		order = range(self.inputs.shape[0])
		np.random.shuffle(order)
		self.inputs = self.inputs[order, :]
		self.outputs = self.outputs[order, :]

	def _forward_prop(self, inputs):
		self.activation[0] = inputs
		# Forward propagation.
		for i in range(self.num_layers - 1):
			layer_input = np.concatenate(
				[np.ones((self.inputs_per_batch,1)), self.sig_activation[i]], axis=1)
			self.activation[i+1] = np.dot(layer_input, self.theta[i].T)
			self.sig_activation[i+1] = self.sigmoid(self.activation[i+1])

	def _back_prop(self, outputs):
		self.delta[self.num_layers-1] = (self.sig_activation[-1] - outputs)
		for i in range(self.num_layers-1, 1, -1):
			self.delta[i-1] = (
				np.dot(self.delta[i], self.theta[i-1])[:, 1:]) * self.grad(self.activation[i-1])

	def _update_theta(self):
		for i in range(self.num_layers-1, 0, -1):
			diff = np.dot(
				self.delta[i].T, np.concatenate(
					[np.ones((self.inputs_per_batch,1)), self.sig_activation[i-1]], axis=1))
			diff = (self.eta * diff) / self.inputs_per_batch
			self.theta[i-1] -= (diff + self.momentum*self.old_updates[i-1])
			self.old_updates[i-1] = diff

	def _perform_single_iter(self, inputs, outputs):
		self._forward_prop(inputs)
		self._back_prop(outputs)
		self._update_theta()

	def _perform_single_batch_iteration(self):
		self._shuffle()
		self._perform_single_iter(self.inputs, self.outputs)
		error = self.calculate_error(self.outputs, self.sig_activation[-1])
		return error

	def _perform_single_mini_batch_iteration(self):
		self._shuffle()
		error = 0
		mini_batch_count = 0
		for i in range(0, self.test_cases, self.mini_batch_size):
			batch_input = self.inputs[i:i+self.mini_batch_size, :]
			batch_output = self.outputs[i:i+self.mini_batch_size, :]

			# Check if remaining number inputs is less than batch size.
			if batch_input.shape[0] < self.mini_batch_size:
				break
			self._perform_single_iter(batch_input, batch_output)
			error += self.calculate_error(batch_output, self.sig_activation[-1])
			mini_batch_count += 1
		return error / mini_batch_count

	def _perform_stochastic_iteration(self):
		error = 0
		self._shuffle()
		#for i in range(self.test_cases):
		self._perform_single_iter(self.inputs[0], self.outputs[0])
		error = self.calculate_error(self.outputs[0], self.sig_activation[-1])
		return error #/ self.test_cases

	def _perform_single_learning_iter(self):
		if self.learning_method == self.BATCH_LEARNING:
			return self._perform_single_batch_iteration()
		elif self.learning_method == self.MINI_BATCH_LEARNING:
			return self._perform_single_mini_batch_iteration()
		elif self.learning_method == self.STOCHASTIC_LEARNING:
			return self._perform_stochastic_iteration()

	def get_outputs(self, inputs):
		for i in range(self.num_layers - 1):
			inputs = np.concatenate(
				[np.ones((inputs.shape[0],1)), inputs], axis=1)
			inputs = self.sigmoid(np.dot(inputs, self.theta[i].T))
		return inputs

	def train(self, max_epoch=None):
		old_validation_error1 = 10003
		old_validation_error2 = 10002
		validation_error = 10001
		error = 1000

		if max_epoch:
			epoch = 1
			while max_epoch >= epoch:
				error = self._perform_single_learning_iter()
				epoch += 1
				if (epoch % 100) == 0:
					print "E(train, %d epoch) = %f" % (epoch, error)

		elif self.early_stopping:
			while ((old_validation_error2 - old_validation_error1) > 0.001 or 
					(old_validation_error1 - validation_error) > 0.001):
				error = self._perform_single_learning_iter()

				old_validation_error2 = old_validation_error1
				old_validation_error1 = validation_error
				# check validation error.
				validation_test = self.get_outputs(self.validation_inputs)
				validation_error = self.calculate_error(
					self.validation_outputs, validation_test)
			print "E(validation) = %f, E(train) = %f" % (validation_error,  error)

