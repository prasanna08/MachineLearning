import abc
from math import exp

"""Classes for annealing learning rate schedules."""

class AnnealingSchedule(object):
	def __init__(self, initial_lr, decay_rate, decay_step):
		self.initial_lr = initial_lr
		self.lr = initial_lr
		self.k = decay_rate
		self.decay_step = decay_step
		self.global_step = 0

	def __mul__(self, z):
		self.global_step += 1
		if (self.global_step % self.decay_step) == 0: self._anneal_lr()
		return self.lr * z

	def __rmul__(self, z):
		return self.__mul__(z)

	@abc.abstractmethod
	def _anneal_lr(self):
		"""Define your annealing schedule here."""
		return


class InvScaling(AnnealingSchedule):
	def __init__(self, initial_lr, decay_rate, decay_step=10):
		super(InvScaling, self).__init__(initial_lr, decay_rate, decay_step)

	def _anneal_lr(self):
		self.lr = self.initial_lr / (1 + self.k * self.global_step)


class ExponentialDecay(AnnealingSchedule):
	def __init__(self, initial_lr, decay_rate, decay_step=10):
		super(ExponentialDecay, self).__init__(initial_lr, decay_rate, decay_step)

	def _anneal_lr(self):
		self.lr = self.initial_lr * exp(-self.k * self.global_step)


class StepDecay(AnnealingSchedule):
	def __init__(self, initial_lr, decay_rate, decay_step=10):
		super(StepDecay, self).__init__(initial_lr, decay_rate, decay_step)

	def _anneal_lr(self):
		self.lr = self.lr - self.decay_rate
