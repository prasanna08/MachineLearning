import numpy as np

class NeuralNet(object):
	BATCH_LEARNING = 'batch'
	MINI_BATCH_LEARNING = 'mini_batch'
	STOCHASTIC_LEARNING = 'stochastic'
	def __init__(
			self, train_inputs, train_outputs, hidden_layers, validation_inputs,
			validation_outputs, eta=0.7, momentum=0.9, early_stopping=True,
			method=BATCH_LEARNING, mini_batch_size=None, outtype='softmax',
			cost_func='log', optimizer='adam', regularization=True,
			regularization_param=1.0, lr_decay=0.0, lr_decay_type='inv',
			beta1=0.9, beta2=0.999, eps=1e-8):
		"""NeuralNet class is used to create neural network classifier.

		Args:
			train_inputs: numpy array. This is array containing inputs to the
				neural network of size (n_samples, n_features).
			train_outputs: numpy array. This is array containg expected outputs
				of neural network of size (n_samples, n_classes). In case of
				regression problem n_class is 1. In classification problem
				n_class is total number of classes.
			hidden_layers: list. This list contains number of neurons in hidden
				layer. These are between input layer and output layer. For e.g. if
				this list is [20, 30] then architecture of entire network is
				n_features -> 20 -> 30 -> n_classes.
			validation_inputs: numpy array. This array consists of validation
				inputs of size (n_validation, n_features). n_validation is size
				of number validaion cases.
			validation_outputs: numpy array. This array consists of validation
				outputs of size (n_validation, n_classes).
			eta: float. This value is learning rate parameter of network.
			momentum: float. Momentum rate of network.
			early_stopping: boolean. If enabled, network uses validation set to determine
				when to stop learning.
			method: str. This specified learning method used by network. It can
				be 'batch', 'stochastic' or 'mini_batch' learning method.
				'batch' referes to batch learning of gradient decent.
				'stochastic' referes to stochastic gradient decent (SGD).
				'mini_batch' is variation of SGD and is in middle of batch
				gradient decent and	purely SGD.
			mini_batch_size: int. This must be specified when using 'mini_batch'
				learning method.
			outtype: str. This specifies what function to use in activation of
				output neurons. As of now it support 'sigmoid' only.
			cost_func: str. This tells which cost function to use. Possibel
				choices are 'mse' (Mean Squared Error) and 'log' (or Cross
				Entropy) error function.
			optimizer: str. What type of optimizer to use. Currently supports
				'momentum', 'nag', 'adam', 'adagrad', 'rmsprop'.
			regularization: bool. This specified whether to use regularization
				or not. L2 regularization is used if this is True.
			regularization_param: float. This is regularization parameter used
				for penalty calculation and weights update.
			lr_decay: float. Hyperparameter used for annealing learning rate.
			lr_decay_type: str. Method used for learning rate decay. Possible
				types are 'inv' and 'exp'.
			beta1: float. Adam optimizer hyper parameter.
			beta2: float. Adam optimizer hyper parameter.
			eps: float. Small constant to avoid division by zero in
				Adam, Adagrad and RMSprop optimizer.
		"""
		# Prepare train_data.
		self.inputs = np.array(train_inputs)
		self.outputs = np.array(train_outputs)
		self.test_cases = self.inputs.shape[0]
		self.input_neurons = np.array([len(self.inputs[0])])
		self.output_neurons = np.array([len(self.outputs[0])])
		self.hidden_layers = np.array(hidden_layers)
		self.eta = eta
		self.momentum = momentum
		self.early_stopping = early_stopping
		self.outtype = outtype
		self.cost_func = cost_func
		self.optimizer = optimizer
		self.regularization = regularization
		self.regularization_param = regularization_param
		self.learning_method = method
		self.lr_decay = lr_decay
		self.lr_decay_type = lr_decay_type
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.init_eta = eta

		# Prepare validation data for early stopping.
		if early_stopping:
			self.validation_inputs = np.array(validation_inputs)
			self.validation_outputs = np.array(validation_outputs)

		if (self.learning_method == self.MINI_BATCH_LEARNING and
				mini_batch_size == None):
			raise Exception(
				"Please enter mini_batch_size if using "
				"MINI_BATCH_LEARNING method")
		self.mini_batch_size = mini_batch_size

		# Initialize all required variables.
		self.neural_layers = np.concatenate(
			[self.input_neurons, self.hidden_layers, self.output_neurons])
		self.num_layers = self.neural_layers.shape[0]
		self.theta = [np.random.randn(self.neural_layers[i+1], self.neural_layers[i]+1) / np.sqrt(2.0 / self.neural_layers[i]+1)
			for i in np.arange(self.num_layers-1)]

		self.old_updates = [np.zeros((self.neural_layers[i+1], self.neural_layers[i]+1))
			for i in np.arange(self.num_layers-1)]
		self.cache = [np.zeros((self.neural_layers[i+1], self.neural_layers[i]+1))
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

	def sigmoid_grad(self, z):
		sigm = self.sigmoid(z)
		return sigm*(1-sigm)

	def softmax(self, z):
		if z.shape[0] > 1:
			return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)
		else:
			return np.exp(z) / np.sum(np.exp(z))

	def softmax_delta(self, target, z):
		delk = np.zeros(z.shape)
		delta = np.zeros(z.shape)
		for i in range(delta.shape[1]):
			delk[:, i] = 1
			delta[:, i] = np.sum((z - target) * z * (delk - z[:, i].reshape(z.shape[0],1)), axis=1)
			delk[:, i] = 0
		return delta

	def calculate_regularization_penalty(self):
		penalty = 0
		for t in self.theta:
			penalty += (self.regularization_param * np.sum(t[:, 1:]**2) / (2 * self.inputs_per_batch))
		return penalty

	def calculate_error(self, target, output):
		reg_penalty = 0
		cost = 0
		if self.regularization == True:
			reg_penalty = self.calculate_regularization_penalty()

		if self.cost_func == 'log':
			cost = np.sum(-(target * np.log(output) + (1 - target) * np.log(1 - output))) / target.shape[0]
		elif self.cost_func == 'mse':
			cost = np.sum((output - target) ** 2)/(2 * output.shape[0])
		else:
			raise Exception("Unknown cost function supplied. Please set it to "
				"mse or log.")
		return cost + reg_penalty

	def calculate_output(self, outputs):
		if self.outtype == 'sigmoid':
			return self.sigmoid(outputs)
		elif self.outtype == 'softmax':
			return self.softmax(outputs)
		else:
			raise Exception("Unknow output activation function.")

	def calculate_output_delta(self, target, outputs):
		if self.cost_func == 'log':
			return (outputs - target)
		elif self.cost_func == 'mse':
			if self.outtype == 'sigmoid':
				return (outputs - target) * outputs * (1 - outputs)
			elif self.outtype == 'softmax':
				return self.softmax_delta(target, outputs)
			else:
				raise Exception("Unknown output activation function.")

	def _shuffle(self):
		order = range(self.inputs.shape[0])
		np.random.shuffle(order)
		self.inputs = self.inputs[order, :]
		self.outputs = self.outputs[order, :]

	def _decay_lr(self):
		if self.lr_decay_type == 'inv':
			self.eta = self.init_eta * (1. / (1 + self.global_step * self.lr_decay))
		elif self.lr_decay_type == 'exp':
			self.eta = self.init_eta * np.exp(-1 * self.lr_decay * self.global_step)
		else:
			raise Exception('Unknown method for learning rate decay.')

	def _forward_prop(self, inputs):
		for i in range(self.num_layers - 1):
			self.sig_activation[i] = (
				inputs if i==0 else self.sigmoid(self.activation[i]))
			layer_input = np.concatenate(
				[np.ones((self.inputs_per_batch,1)), self.sig_activation[i]], axis=1)
			self.activation[i+1] = np.dot(layer_input, self.theta[i].T)

		self.sig_activation[-1] = self.calculate_output(self.activation[-1])

	def _back_prop(self, outputs):
		self.delta[-1] = self.calculate_output_delta(outputs, self.sig_activation[-1])

		for i in range(self.num_layers-1, 1, -1):
			self.delta[i-1] = (
				np.dot(self.delta[i], self.theta[i-1])[:, 1:]) * self.sig_activation[i-1] * (1 - self.sig_activation[i-1])

	def _update_theta(self):
		for i in range(self.num_layers-1, 0, -1):
			grad = np.dot(
				self.delta[i].T, np.concatenate(
					[np.ones((self.inputs_per_batch,1)), self.sig_activation[i-1]], axis=1))
			grad = grad / self.inputs_per_batch

			if self.regularization:
				t = np.zeros(self.theta[i-1].shape)
				t[:, 1:] = self.theta[i-1][:, 1:]
				grad += (self.regularization_param * t / self.inputs_per_batch)

			grad = self.optimize(i, grad)
			self.theta[i-1] += grad

	def optimize(self, layer, grad):
		if self.optimizer == 'vanilla':
			return -1 * (self.eta * grad)
		elif self.optimizer == 'momentum':
			return self.momentum_optimizer(layer, grad)
		elif self.optimizer == 'nag':
			return self.nag_optimizer(layer, grad)
		elif self.optimizer == 'adam':
			return self.adam_optimizer(layer, grad)
		elif self.optimizer == 'adagrad':
			return self.adagrad_optimizer(layer, grad)
		elif self.optimizer == 'rmsprop':
			return self.rmsprop_optimizer(layer, grad)
		else:
			raise Exception('Unknown optimizer.')

	def momentum_optimizer(self, layer, grad):
		new_grad = self.momentum * self.old_updates[layer - 1] - self.eta * grad
		self.old_updates[layer - 1] = new_grad
		return new_grad

	def nag_optimizer(self, layer, grad):
		new_grad = self.momentum_optimizer(layer, grad)
		return self.momentum * new_grad - self.eta * grad

	def adagrad_optimizer(self, layer, grad):
		self.cache[layer - 1] += (grad**2)
		return -1 * (self.eta * grad) / (np.sqrt(self.cache[layer - 1]) + self.eps)

	def rmsprop_optimizer(self, layer, grad):
		self.cache[layer - 1] = self.beta1 * self.cache[layer - 1] + (1 - self.beta1) * (grad**2)
		return -1 * (self.eta * grad) / (np.sqrt(self.cache[layer - 1]) + self.eps)

	def adam_optimizer(self, layer, grad):
		self.old_updates[layer - 1] = self.beta1 * self.old_updates[layer - 1] + (1 - self.beta1) * grad
		self.cache[layer - 1] = self.beta2 * self.cache[layer - 1] + (1 - self.beta2) * (grad**2)
		bias_correction_grad = (1.0 / (1 - self.beta1 ** self.global_step))
		bias_correction_cache = (1.0 / (1 - self.beta2 ** self.global_step))
		return -1 * (self.eta * self.old_updates[layer - 1] * bias_correction_grad) / (np.sqrt(self.cache[layer - 1] * bias_correction_cache) + self.eps)

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
		for i in range(self.test_cases):
			ins = self.inputs[i].reshape(1, self.inputs.shape[1])
			outs = self.outputs[i].reshape(1, self.outputs.shape[1])
			self._perform_single_iter(ins, outs)
			error += self.calculate_error(outs, self.sig_activation[-1])
		return error / self.test_cases

	def _perform_single_learning_iter(self):
		if self.learning_method == self.BATCH_LEARNING:
			return self._perform_single_batch_iteration()
		elif self.learning_method == self.MINI_BATCH_LEARNING:
			return self._perform_single_mini_batch_iteration()
		elif self.learning_method == self.STOCHASTIC_LEARNING:
			return self._perform_stochastic_iteration()
		else:
			raise Exception("Unknown learning method. Please set learning "
				"to batch, mini_batch ot stochastic.")

	def get_outputs(self, inputs):
		for i in range(self.num_layers - 1):
			inputs = inputs if i==0 else self.sigmoid(activations)
			inputs = np.concatenate(
				[np.ones((inputs.shape[0],1)), inputs], axis=1)
			activations = np.dot(inputs, self.theta[i].T)
		return self.calculate_output(activations)

	def get_predicted_classes(self, inputs):
		outputs = self.get_outputs(inputs)
		return outputs.argmax(axis=1)

	def calculate_accuracy(self, inputs, target):
		return np.sum(self.get_predicted_classes(inputs) == target) / float(inputs.shape[0]) * 100.0

	def _calculate_accuracy(self, output, target):
		return np.sum(output.argmax(axis=1) == target.argmax(axis=1)) / 100.0

	def train(self, max_epoch=None, report_back_at=100, show_validation_error=True):
		old_validation_error1 = 10003
		old_validation_error2 = 10002
		validation_error = 10001
		error = 1000
		self.global_step = 1

		if max_epoch:
			while max_epoch >= self.global_step:
				self.global_step += 1
				error = self._perform_single_learning_iter()
				self._decay_lr()
				if (self.global_step % report_back_at) == 0:
					print "E(train, %d epoch) = %f" % (self.global_step, error)
					if show_validation_error:
						validation_test = self.get_outputs(self.validation_inputs)
						validation_error = self.calculate_error(
							self.validation_outputs, validation_test)
						print "E(validation) = %f" % validation_error

		elif self.early_stopping:
			while ((old_validation_error2 - old_validation_error1) > 0.0001 or 
					(old_validation_error1 - validation_error) > 0.0001):
				self.global_step += 1
				error = self._perform_single_learning_iter()
				self._decay_lr()
				
				old_validation_error2 = old_validation_error1
				old_validation_error1 = validation_error

				validation_test = self.get_outputs(self.validation_inputs)
				validation_error = self.calculate_error(
					self.validation_outputs, validation_test)
			print "E(validation) = %f, E(train) = %f" % (validation_error,  error)

		else:
			print ("Unknown learning stopping method. Please set early_stopping "
				"to True, or provide epoch count while train()")
