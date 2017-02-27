"""Contractive auto encoder."""

import numpy as np
import preprocess

class BatchGenerator(object):
	def __init__(self, batch_size, data):
		self.batch_size = batch_size
		self._cursor = 0
		self.data_length = data.shape[0]
		self.data = data

	def next_batch(self):
		old_cursor = self._cursor
		self._cursor = (self._cursor + 1) % (self.data_length - self.batch_size)
		return self.data[old_cursor: old_cursor+self.batch_size, :]


class CAE(object):
	def __init__(self, input_size, encode_size, contractive_factor):
		self.w_encode = preprocess.xavier_init(
			(input_size, encode_size), input_size, encode_size)
		self.b_encode = np.zeros(encode_size)
		self.w_decode = preprocess.xavier_init(
			(encode_size, input_size), encode_size, input_size)
		self.b_decode = np.zeros(input_size)
		self.contractive_factor = contractive_factor

	def Sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def Forward(self, X):
		h = self.Sigmoid(np.dot(X, self.w_encode) + self.b_encode)
		# using sigmoidal output units.
		recons = self.Sigmoid(np.dot(h, self.w_decode) + self.b_decode)

		cache = {
			'X': X,
			'h': h,
			'recons': recons
		}

		return cache

	def Backward(self, d_recons, cache):
		X = cache['X']
		h = cache['h']
		d_hidden = np.dot(d_recons, self.w_decode.T)
		d_w_decode = np.dot(h.T, d_recons)
		d_b_decode = np.sum(d_recons, 0)
		d_linear_h = d_hidden * (h * (1 - h))
		d_w_encode_p1 = np.dot(X.T, d_linear_h)
		d_b_encode = np.sum(d_linear_h, 0)

		# Calculating d_w_encode due to contractive penalty.
		g = ((h) * (1 - h))**2
		p1 = g.sum(0) * self.w_encode
		p2 = np.dot(X.T, g*(1 - 2*h))*(self.w_encode**2).sum(0)
		d_w_encode = d_w_encode_p1 + self.contractive_factor * (p1 + p2)

		d_cache = {
			'd_w_encode': d_w_encode,
			'd_w_decode': d_w_decode,
			'd_b_encode': d_b_encode,
			'd_b_decode': d_b_decode
		}

		return d_cache

	def Train(self, data, batch_size, max_epoch, learning_rate):
		gen = BatchGenerator(batch_size, data)

		for epoch in xrange(max_epoch):
			batch = gen.next_batch()
			cache = self.Forward(batch)
			error, d_recons = self.calculate_error(cache)
			d_cache = self.Backward(d_recons, cache)

			d_w_encode = d_cache['d_w_encode']
			d_w_decode = d_cache['d_w_decode']
			d_b_encode = d_cache['d_b_encode']
			d_b_decode = d_cache['d_b_decode']

			# Vanilla update rule.
			self.w_encode = self.w_encode - learning_rate * (d_w_encode / batch_size)
			self.b_encode = self.b_encode - learning_rate * (d_b_encode / batch_size)
			self.w_decode = self.w_decode - learning_rate * (d_w_decode / batch_size)
			self.b_decode = self.b_decode - learning_rate * (d_b_decode / batch_size)

			print "Epoch: %d, Error: %.5f" % (epoch, error)

	def calculate_error(self, cache):
		recons = cache['recons']
		X = cache['X']
		h = cache['h']
		error = np.sum((recons - X)**2)
		grad = h * (1 - h)
		jacob = np.dot(grad**2, (self.w_encode**2).sum(0))
		contractive_error = np.sum(jacob) * self.contractive_factor / 2
		error = (error + contractive_error) / (2 * X.shape[0])
		d_recons = (recons - X) * recons * (1 - recons)
		return error, d_recons
