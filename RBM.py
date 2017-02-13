"""Implementation of RBM network.

TODO:
	Implement mini batch based training approach.
	Implement output classification layer for supervised training and
		classification.
"""
import numpy as np

class RBM(object):
	def __init__(self, data, n_hidden, nCD, lr=0.1, momentum=0.6, decay=1e-3):
		"""The RBM class.

		Args:
			data: matrix. The training data for RBM.
			n_hidden: int. Number of hidden units.
			nCD: int. Number of steps for gibbs sampling.
			lr: float. Learning rate.
			momentum: float. Momentum rate for Momentum based SGD update.
			decay: float. L2 weight decay coefficient.
		"""
		self.data = data
		self.n_visible = data.shape[1]
		self.n_hidden = n_hidden
		self.nCD = nCD
		self.weights = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_hidden, self.n_visible))
		self.v_bias = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_visible))
		self.h_bias = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_hidden))
		self.lr = lr
		self.momentum = momentum
		self.decay = decay

	def sigm(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def get_h_given_v(self, visible):
		hiddenprob = self.sigm(np.dot(visible, self.weights.T) + self.h_bias)
		hiddenact = (hiddenprob>np.random.uniform(size=(hiddenprob.shape[0], self.n_hidden))).astype('float')
		return hiddenprob, hiddenact

	def get_v_given_h(self, hidden):
		visibleprob = self.sigm(np.dot(hidden, self.weights) + self.v_bias)
		visibleact = (visibleprob>np.random.uniform(size=(visibleprob.shape[0], self.n_visible))).astype('float')
		return visibleprob, visibleact

	def train(self, epoch, persistent):
		"""Train method for training RBM.

		Args:
			epoch: int. Number of training epochs.
			persistent: vector / matrix of size (n_rows, n_hidden). Persistent
				hidden activations to be used for PCD training.
				n_rows is devided by user.
		"""
		dw = 0
		dvb = 0
		dhb = 0
		for e in epoch:
			hiddenp, hiddena = self.get_h_given_v(self.data)

			positive = np.dot(hiddena.T, self.data)
			positivevb = np.sum(self.data, axis=0)
			positivehb = np.sum(hiddenp, axis=0)

			if persistent is not None:
				hiddena = persistent

			for step in range(self.nCD):
				recnsp, recnsa = self.get_v_given_h(hiddena)
				hiddenp, hiddena = self.get_h_given_v(recnsp)

			negative = np.dot(hiddenp.T, recnsp)
			negativevb = recnsp.sum(axis=0)
			negativehb = hiddenp.sum(axis=0)

			dw = (self.lr * ((positive - negative) / self.data.shape[0]) - self.decay * self.weights) + self.momentum * dw
			self.weights += dw
			dvb = (self.lr * (positivevb - negativevb) / self.data.shape[0]) + self.momentum * dvb
			self.v_bias += dvb
			dhb = (self.lr * (positivehb - negativehb) / self.data.shape[0]) + self.momentum * dhb
			self.h_bias += dhb

			error = np.sum((self.data - recnsa)**2)
			print error

	def energy(self, visible, hidden):
		vb = (visible * self.v_bias).sum()
		hb = (hidden * self.h_bias).sum()
		vwh = (np.dot(visible, self.weights.T) * hidden).sum()
		return - (vb + hb + vwh)

	def sample(n_samples, data=None):
		samples = np.zeros((n_samples, self.n_visible))
		if data is None:
			data = np.random.unforom(size=(1, self.n_visible))

		for i in range(n_samples):
			hiddenp, hiddena = self.get_h_given_v(data)
			visiblep, visiblea = self.get_v_given_h(hiddena)
			samples[i, :] = visiblea
			data = visiblea

		return samples
