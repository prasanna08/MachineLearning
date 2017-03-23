import abc
import numpy as np
import model

class optimizer(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, model, batch_size, learning_rate):
		self.global_step = 0
		self.model = model
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.params = model.get_params_mapping()
		self.mappings = {k: v[0] for k, v in self.params.iteritems()}

	def train(self, data, labels=None, max_epochs=100, display_count=10):
		"""Minibatch training method."""
		supervised = isinstance(self.model, model.SupervisedModel)
		print "Obtaining generator"
		if supervised:
			gen = self.model.get_batch_generator(self.batch_size, data, labels)
		else:
			gen = self.model.get_batch_generator(self.batch_size, data)

		print "Starting training"
		for epoch in range(max_epochs):
			average_loss = 0

			# Batch iterator.
			for batch in range(0, len(data), self.batch_size):
				if supervised:
					# If supervised learning.
					batch_input, batch_output = gen.next_batch()
					params, d_params, loss = self.model.train(batch_input, batch_output)
				else:
					# If unsupervised learning.
					batch_input = gen.next_batch()
					params, d_params, loss = self.model.train(batch_input)

				self.global_step += 1
				self._update_params(params, d_params)
				average_loss += loss

				if (self.global_step % display_count) == 0:
					average_loss = average_loss / self.batch_size
					print "Loss at step %d is %.4f" % (self.global_step, average_loss)
					average_loss = 0

	@abc.abstractmethod
	def _update_params(self, params, d_params):
		"""To be implemented by each child class."""
		return


class SGDMinibatch(optimizer):
	def __init__(self, model, batch_size, learning_rate):
		super(SGDMinibatch, self).__init__(model, batch_size, learning_rate)

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			param -= self.learning_rate * gparam


class SGDMomentum(optimizer):
	def __init__(self, model, batch_size, learning_rate, momentum):
		super(SGDMomentum, self).__init__(model, batch_size, learning_rate)
		self.momentum = momentum
		self.old_cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			self.old_cache[k] = self.momentum * self.old_cache[k] - self.learning_rate * gparam
			param -= self.old_cache[k]


class SGDNag(optimizer):
	def __init__(self, model, batch_size, learning_rate, momentum):
		super(SGDNag, self).__init__(model, batch_size, learning_rate)
		self.momentum = momentum
		self.old_cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			self.old_cache[k] = self.momentum * self.old_cache[k] - self.learning_rate * gparam
			param -= (self.momentum * self.old_cache[k] - self.learning_rate * gparam)


class Adagrad(optimizer):
	def __init__(self, model, batch_size, learning_rate, eps=1e-6):
		super(Adagrad, self).__init__(model, batch_size, learning_rate)
		self.eps = eps
		self.g2cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			self.g2cache[k] += gparam**2
			lr = self.learning_rate / (np.sqrt(self.g2cache[k]) + self.eps)
			param -= lr * gparam


class RMSProp(optimizer):
	def __init__(self, model, batch_size, learning_rate, beta1=0.9, eps=1e-6):
		super(RMSProp, self).__init__(model, batch_size, learning_rate)
		self.eps = eps
		self.beta1 = beta1
		self.g2cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			self.g2cache[k] = self.beta1 * self.g2cache[k] + (1 - self.beta1) * gparam**2
			lr = self.learning_rate / (np.sqrt(self.g2cache[k]) + self.eps)
			param -= lr * gparam


class Adam(optimizer):
	def __init__(
			self, model, batch_size, learning_rate, beta1=0.9, beta2=0.95,
			eps=1e-6):
		super(Adam, self).__init__(model, batch_size, learning_rate)
		self.eps = eps
		self.beta1 = beta1
		self.beta2 = beta2
		self.old_cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}
		self.g2cache = {k: np.zeros(v[1]) for k, v in self.params.iteritems()}

	def _update_params(self, params, d_params):
		for k, v in self.mappings.iteritems():
			param = params[k]
			gparam = d_params[v] / self.batch_size
			self.old_cache[k] = self.beta1 * self.old_cache[k] + (1 - self.beta1) * gparam
			self.g2cache[k] = self.beta2 * self.g2cache[k] + (1 - self.beta2) * gparam**2
			bias_correction_grad = 1.0  / (1.0 - self.beta1 ** self.global_step)
			bias_correction_g2cache = 1.0  / (1.0 - self.beta2 ** self.global_step)
			param -= (self.learning_rate * self.old_cache[k] * bias_correction_grad) \
				/ (np.sqrt(self.g2cache[k] * bias_correction_g2cache) + self.eps)
