import abc

class Model(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def get_params_mapping(self):
		"""This method should return a dictionary mapping parameter in cache
		to its gradient in d_cache and shape of the parameter matrix."""
		return

class SupervisedModel(Model):
	@abc.abstractmethod
	def get_batch_generator(self, batch_size, data, labels):
		"""Returns a dictionary of parameters for batch generator class specific
		to given model.
		"""
		return

	@abc.abstractmethod
	def train(self, data, labels):
		"""This method should perform forward and backward propogation for given
		data and return cache, d_cache and loss."""
		return


class UnsupervisedModel(Model):
	@abc.abstractmethod
	def get_batch_generator(self, batch_size, data):
		"""Returns a dictionary of parameters for batch generator class specific
		to given model.
		"""
		return

	@abc.abstractmethod
	def train(self, data):
		"""This method should perform forward and backward propogation for given
		data and return cache, d_cache and loss."""
		return