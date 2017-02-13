"""Implementation of RBM network."""
import numpy as np

class RBM(object):
	def __init__(self, data, n_hidden, nCD, lr):
		"""The RBM class.

		Args:
			data: matrix. The training data for RBM.
			n_hidden: int. Number of hidden units.
			nCD: int. Number of steps for gibbs sampling.
			lr: float. Learning rate.
		"""
		self.data = data
		self.n_visible = data.shape[1]
		self.n_hidden = n_hidden
		self.nCD = nCD
		self.weights = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_hidden, self.n_visible))
		self.v_bias = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_visible))
		self.h_bias = np.ranodm.uniform(low=0.0, high=0.1, size=(self.n_hidden))

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
			persistent: vector of size n_hidden. Persistent hidden activations
				to be used for PCD training.
		"""
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

			dw = lr * (positive - negative) / self.data.shape[0]
			self.weights += dw
			dvb = lr * (positivevb - negativevb) / self.data.shape[0]
			self.v_bias += dvb
			dhb = lr * (positivehb - negativehb) / self.data.shape[0]
			self.h_bias += dhb

	def energy(self, visible, hidden):
		vb = (visible * self.v_bias).sum()
		hb = (hidden * self.h_bias).sum()
		vwh = (np.dot(visible, self.weights.T) * hidden).sum()
		return - (vb + hb + vwh)
