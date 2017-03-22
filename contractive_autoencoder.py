import batch_generators
import model
import numpy as np
import preprocess

"""Contractive autoencoder."""

class CAE(model.UnsupervisedModel):
	def __init__(self, input_size, encode_size, contractive_factor):
		self.w_encode = preprocess.xavier_init((input_size, encode_size))
		self.b_encode = np.zeros(encode_size)
		self.w_decode = preprocess.xavier_init((encode_size, input_size))
		self.b_decode = np.zeros(input_size)
		self.contractive_factor = contractive_factor
		self.params = self.get_params()

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def get_params_mapping(self):
		mapper = {
			'w_encode': ['d_w_encode', self.w_encode.shape],
			'w_decode': ['d_w_decode', self.w_decode.shape],
			'b_encode': ['d_b_encode', self.b_encode.shape],
			'b_decode': ['d_b_decode', self.b_decode.shape]
		}

		return mapper

	def get_params(self):
		params = {
			'w_encode': self.w_encode,
			'w_decode': self.w_decode,
			'b_encode': self.b_encode,
			'b_decode': self.b_decode
		}

		return params

	def get_batch_generator(self, batch_size, data):
		return batch_generators.EncoderBatchGenerator(batch_size, data)

	def forward(self, X):
		h = self.sigmoid(np.dot(X, self.w_encode) + self.b_encode)
		# using sigmoidal output units.
		recons = self.sigmoid(np.dot(h, self.w_decode) + self.b_decode)

		cache = {
			'X': X,
			'h': h,
			'recons': recons
		}

		return cache

	def backward(self, d_recons, cache):
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

	def compute_loss_and_gradient(self, cache):
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

	def train(self, data):
		cache = self.forward(data)
		loss, dout = self.compute_loss_and_gradient(cache)
		d_params = self.backward(dout, cache)
		return self.params, d_params, loss
