"""Implementation of RBM network.

TODO:
	Implement mini batch based training approach.
"""
import numpy as np

class RBM(object):
	def __init__(self, data, n_hidden, labels=None, lr=0.3, momentum=0.6, decay=1e-3):
		"""The RBM class.

		Args:
			data: matrix. The training data for RBM.
			n_hidden: int. Number of hidden units.
			labels: matrix. Labels for data for supervised training.
			lr: float. Learning rate.
			momentum: float. Momentum rate for Momentum based SGD update.
			decay: float. L2 weight decay coefficient.
		"""
		self.data = data
		self.n_visible = data.shape[1]
		self.n_hidden = n_hidden
		self.weights = np.random.uniform(low=0.0, high=0.1, size=(self.n_visible, self.n_hidden))
		self.v_bias = np.random.uniform(low=0.0, high=0.1, size=(self.n_visible))
		self.h_bias = np.random.uniform(low=0.0, high=0.1, size=(self.n_hidden))
		self.lr = lr
		self.momentum = momentum
		self.decay = decay
		self.labels = labels
		if labels is not None:
			self.n_labels = labels.shape[1]
			self.labelweights = np.random.uniform(low=0.0, high=0.1, size=(self.n_hidden, self.n_labels))
			self.labelbias = np.random.uniform(low=0.0, high=0.1, size=(self.n_labels))

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def softmax(self, z):
		if z.ndim > 1:
			z -= z.max(axis=1)[:, np.newaxis]
			ez = np.exp(z)
			ez = ez / ez.sum(axis=1)[:, np.newaxis]
		else:
			z -= z.max()
			ez = np.exp(z)
			ez = ez / ez.sum()

		return ez

	def get_h_given_v(self, visible, labels):
		if labels is not None:
			hiddenprob = self.sigmoid(np.dot(visible, self.weights) + self.h_bias + np.dot(labels, self.labelweights.T))
		else:
			 hiddenprob = self.sigmoid(np.dot(visible, self.weights) + self.h_bias)
		hiddenact = (hiddenprob>np.random.uniform(size=(hiddenprob.shape[0], self.n_hidden))).astype(np.int32)
		return hiddenprob, hiddenact

	def get_v_given_h(self, hidden):
		visibleprob = self.sigmoid(np.dot(hidden, self.weights.T) + self.v_bias)
		visibleact = (visibleprob>np.random.uniform(size=(visibleprob.shape[0], self.n_visible))).astype(np.int32)
		labelsprob = None
		if self.labels is not None:
			labels = self.sigmoid(np.dot(hidden, self.labelweights) + self.labelbias)
			labelsprob = self.softmax(labels)
		return visibleprob, visibleact, labelsprob

	def train(self, epoch, nCD, PCD=False, display_at=50):
		"""Train method for training RBM.

		Args:
			epoch: int. Number of training epochs.
			nCD: int. Number of alternating gibbs sampling steps.
			PCD: bool. If True then use PCD training else CD training.
		"""
		dw = 0
		dvb = 0
		dhb = 0
		dlabelw = 0
		dlabelb = 0

		if PCD:
			persistent = (np.random.uniform(size=self.data.shape) > np.random.randn(*self.data.shape)).astype(np.int32)

		for e in range(epoch):
			hiddenp, hiddena = self.get_h_given_v(self.data, self.labels)
			labelsp = self.labels

			positive = np.dot(self.data.T, hiddenp)
			positivevb = np.sum(self.data, axis=0)
			positivehb = np.sum(hiddenp, axis=0)

			if self.labels is not None:
				positivelabels = np.dot(hiddenp.T, labelsp)
				positivelabelsb = labelsp.sum(axis=0)

			if PCD:
				recnsp = persistent
				for step in range(nCD):
					hiddenp, hiddena = self.get_h_given_v(recnsp, labelsp)
					recnsp, recnsa, labelsp = self.get_v_given_h(hiddena)
				persistent = recnsp
			else:
				for step in range(nCD):
					recnsp, recnsa, labelsp = self.get_v_given_h(hiddena)
					hiddenp, hiddena = self.get_h_given_v(recnsp, labelsp)

			if self.labels is not None:
				negativelabels = np.dot(hiddenp.T, labelsp)
				negativelabelsb = labelsp.sum(axis=0)
				dlabelw = (self.lr * (positivelabels - negativelabels) / self.data.shape[0]) - self.decay * self.labelweights + self.momentum * dlabelw
				self.labelweights += dlabelw
				dlabelb = (self.lr * (positivelabelsb - negativelabelsb) / self.data.shape[0]) + self.momentum * dlabelb
				self.labelbias += dlabelb

			negative = np.dot(recnsp.T, hiddenp)
			negativevb = recnsp.sum(axis=0)
			negativehb = hiddenp.sum(axis=0)

			dw = (self.lr * (positive - negative) / self.data.shape[0]) - self.decay * self.weights + self.momentum * dw
			self.weights += dw
			dvb = (self.lr * (positivevb - negativevb) / self.data.shape[0]) + self.momentum * dvb
			self.v_bias += dvb
			dhb = (self.lr * (positivehb - negativehb) / self.data.shape[0]) + self.momentum * dhb
			self.h_bias += dhb

			error = np.sum((self.data - recnsa)**2) / self.data.shape[0]
			if e % display_at == 0:
				print error, np.mean(self.energy(self.data, hiddena, self.labels))

	def energy(self, visible, hidden, labels):
		vb = np.dot(visible, self.v_bias)
		hb = np.dot(hidden, self.h_bias)
		vwh = (np.dot(visible, self.weights) * hidden).sum(axis=1)
		lbe = 0
		if labels is not None:
			lb = np.dot(labels, self.labelbias)
			llwh = (np.dot(hidden, self.labelweights) * labels).sum(axis=1)
			lbe = lb + llwh
		return -(vb + hb + vwh + lbe)

	def sample(self, n_samples, data=None):
		samples = np.zeros((n_samples, self.n_visible))
		if data is None:
			data = np.random.uniform(size=(1, self.n_visible))

		for i in range(n_samples):
			hiddenp, hiddena = self.get_h_given_v(data, None)
			visiblep, visiblea, labelsp = self.get_v_given_h(hiddena)
			samples[i, :] = visiblea
			data = visiblea

		return samples

	def classify(self, visible, labels, n_samples=1):
		for i in range(n_samples):
			hiddenp, hiddena = self.get_h_given_v(visible, labels)
			visible, recnsa, labelsp = self.get_v_given_h(hiddena)
		return labelsp.argmax(axis=1)

	def free_energy(self, visible):
		wx_b = np.dot(visible, self.weights) + self.h_bias
		vbias_term = np.dot(visible, self.v_bias)
		hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
		return -(hidden_term + vbias_term)
