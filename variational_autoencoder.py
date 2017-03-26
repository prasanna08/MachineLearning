import batch_generators
import model
import numpy as np
import preprocess

"""Variational autoencoder."""

class VAE(model.UnsupervisedModel):
	def __init__(self, n_input, n_hidden, n_z):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_z = n_z
		self.n_output = n_input

		# Initialize parameters.
		self.w_enc = preprocess.xavier_init((n_input, n_hidden))
		self.b_enc = np.zeros(n_hidden)
		self.w_mean = preprocess.xavier_init((n_hidden, n_z))
		self.b_mean = np.zeros(n_z)
		self.w_var = preprocess.xavier_init((n_hidden, n_z))
		self.b_var = np.zeros(n_z)
		self.w_dec = preprocess.xavier_init((n_z, n_hidden))
		self.b_dec = np.zeros(n_hidden)
		self.w_out = preprocess.xavier_init((n_hidden, n_input))
		self.b_out = np.zeros(n_input)

		self.params = self.get_params()

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def get_batch_generator(self, batch_size, data):
		return batch_generators.EncoderBatchGenerator(batch_size, data)

	def get_params_mapping(self):
		mapper = {
			'w_enc': ['d_w_enc', self.w_enc.shape],
			'b_enc': ['d_b_enc', self.b_enc.shape],
			'w_mean': ['d_w_mean', self.w_mean.shape],
			'b_mean': ['d_b_mean', self.b_mean.shape],
			'w_var': ['d_w_var', self.w_var.shape],
			'b_var': ['d_b_var', self.b_var.shape],
			'w_dec': ['d_w_dec', self.w_dec.shape],
			'b_dec': ['d_b_dec', self.b_dec.shape],
			'w_out': ['d_w_out', self.w_out.shape],
			'b_out': ['d_b_out', self.b_out.shape]
		}

		return mapper

	def get_params(self):
		params = {
			'w_enc': self.w_enc,
			'b_enc': self.b_enc,
			'w_mean': self.w_mean,
			'b_mean': self.b_mean,
			'w_var': self.w_var,
			'b_var': self.b_var,
			'w_dec': self.w_dec,
			'b_dec': self.b_dec,
			'w_out': self.w_out,
			'b_out': self.b_out
		}

		return params

	def forward(self, x):
		# Forward pass of encoder.
		hidden_enc = np.dot(x, self.w_enc) + self.b_enc
		t_hidden_enc = np.tanh(hidden_enc)
		mean = np.dot(t_hidden_enc, self.w_mean) + self.b_mean
		log_var2 = np.dot(t_hidden_enc, self.w_var) + self.b_var
		dev = np.sqrt(np.exp(log_var2))

		# Sample z.
		eps = np.random.normal(size=mean.shape, loc=0, scale=1)
		z = mean + dev * eps

		# Forward pass of decoder.
		hidden_dec = np.dot(z, self.w_dec) + self.b_dec
		t_hidden_dec = np.tanh(hidden_dec)
		y = self.sigmoid(np.dot(t_hidden_dec, self.w_out) + self.b_out)

		cache = {
			'x': x,
			't_hidden_enc': t_hidden_enc,
			'mean': mean,
			'log_var2': log_var2,
			'eps': eps,
			'z': z,
			't_hidden_dec': t_hidden_dec,
			'y': y
		}

		return cache

	def backward(self, dout, cache):
		# Get forward prop states.
		x = cache['x']
		t_hidden_enc = cache['t_hidden_enc']
		mean = cache['mean']
		log_var2 = cache['log_var2']
		eps = cache['eps']
		z = cache['z']
		t_hidden_dec = cache['t_hidden_dec']
		dev = np.sqrt(np.exp(log_var2))

		# Backprop through decoder.
		d_t_hidden_dec = np.dot(dout, self.w_out.T)
		d_hidden_dec = d_t_hidden_dec * (1 - (t_hidden_dec ** 2))
		d_z = np.dot(d_hidden_dec, self.w_dec.T)
		d_mean = d_z.copy()
		d_dev = d_z * eps
		d_log_var2 = d_dev * dev / 2

		# Backprop through KL divergence error.
		d_mean += mean
		d_log_var2 += (np.exp(log_var2) - 0.5)

		# Backprop through encoder.
		d_t_hidden_enc = np.dot(d_mean, self.w_mean.T)
		d_t_hidden_enc += np.dot(d_log_var2, self.w_var.T)
		d_hidden_enc = d_t_hidden_enc * (1 - (t_hidden_enc ** 2))

		# Gradient of various params.
		d_w_enc = np.dot(x.T, d_hidden_enc)
		d_b_enc = d_hidden_dec.sum(axis=0)

		d_w_mean = np.dot(t_hidden_enc.T, d_mean)
		d_b_mean = d_mean.sum(axis=0)

		d_w_var = np.dot(t_hidden_enc.T, d_log_var2)
		d_b_var = d_log_var2.sum(axis=0)

		d_w_dec = np.dot(z.T, d_hidden_dec)
		d_b_dec = d_hidden_dec.sum(axis=0)

		d_w_out = np.dot(t_hidden_dec.T, dout)
		d_b_out = dout.sum(axis=0)

		d_cache = {
			'd_w_enc': d_w_enc,
			'd_b_enc': d_b_enc,
			'd_w_mean': d_w_mean,
			'd_b_mean': d_b_mean,
			'd_w_var': d_w_var,
			'd_b_var': d_b_var,
			'd_w_dec': d_w_dec,
			'd_b_dec': d_b_dec,
			'd_w_out': d_w_out,
			'd_b_out': d_b_out
		}

		return d_cache

	def compute_loss_and_gradient(self, cache):
		# Get forward prop states.
		x = cache['x']
		y = cache['y']
		mean = cache['mean']
		log_var2 = cache['log_var2']
		dev = np.sqrt(np.exp(log_var2))

		# Calculate Bernoulli loss.
		loss = x * np.log(y) + (1 - x) * np.log(1 - y)

		# Calculate KL loss.
		kl_loss = (mean**2 + dev**2 - 1 - log_var2) / 2

		# Calculate total loss.
		total_loss = loss.sum()
		total_kl_loss = kl_loss.sum()
		total_loss = -1 * (total_loss - total_kl_loss)

		# Calculate gradient of output.
		dout = (y - x)

		return total_loss, dout

	def train(self, data):
		cache = self.forward(data)
		loss, dout = self.compute_loss_and_gradient(cache)
		d_params = self.backward(dout, cache)
		return self.params, d_params, loss
