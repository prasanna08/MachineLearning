"""This is word2vec model for high quality representation of words."""
import numpy as np
import tensorflow as tf
import preprocess

def read_file(file_name):
	f = open(file_name, 'r')
	words = f.read().split()


def word2id(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocabulary_size - 1))
	word2ids = dict()
	
	for word, _ in count:
		word2ids[word] = len(word2ids)

	unk_count = 0
	indexed_data = []
	for word in words:
		if word in word2ids:
			index = word2ids[word]
		else:
			index = 0
			unk_count += 1
		indexed_data.append(index)

	ids2words = dict(zip(word2ids.values(), word2ids.keys()))
	return word2ids, ids2words, count, indexed_data


class BatchGenerator(object):
	def __init__(
			self, batch_size, window_size, n_skips, indexed_data,
			skip_gram=True):
		self.data = indexed_data
		self.window_size = window_size
		self.n_skips = n_skips
		self.batch_size = batch_size
		self._cursor = 0

		if skip_gram:
			self.next_batch = self.next_batch_skip
		else:
			self.next_batch = self.next_batch_cbow

		# batch_size should be multiple of n_skips because each target word
		# is present in batch for n_skips times.
		assert batch_size % n_skips == 0

	def next_batch_cbow(self):
		batch_input = np.ndarray(
			shape=(self.batch_size, self.n_skips), dtype=np.int32)
		batch_output = np.ndarray(shape=(self.batch_size), dtype=np.int32)

		k = 0
		for _ in range(self.n_skips):
			output, inp = self.next_batch_skip()
			for i in range(0, self.batch_size, self.n_skips):
				batch_input[k] = inp[i:i+self.n_skips]
				batch_output[k] = output[i]
				k += 1

		return batch_input, batch_output

	def next_batch_skip(self):
		batch_input = np.ndarray(shape=(self.batch_size), dtype=np.int32)
		batch_output = np.ndarray(shape=(self.batch_size), dtype=np.int32)

		words = self.data[self._cursor: self._cursor + (2 * self.window_size) + 1]
		#cursor = self._cursor
		#for i in range(2 * self.window_size + 1):
		#	words.append(self.data[cursor])
		#	cursor = (cursor + 1) % len(self.data)

		for i in range(self.batch_size // self.n_skips):
			target = words[self.window_size]

			order = range(len(words))
			order.remove(self.window_size)
			np.random.shuffle(order)
			
			labels = [words[x] for x in order[:self.n_skips]]
			# insert data in batch_input and output arrays.
			for k in range(self.n_skips):
				batch_input[i*self.n_skips + k] = target
				batch_output[i*self.n_skips + k] = labels[k]

			self._cursor = (self._cursor + 1) % len(self.data)
			words.pop(0)
			words.append(self.data[self._cursor])

		return batch_input, batch_output


class Word2Vec(object):
	def __init__(
			self, batch_size, window_size, n_skips, vocabulary_size,
			embedding_dim, neg_k=10):
		"""Word2Vec class.

		It is assumed that data provided is numerical and contains word_ids rather
		than actual words. Currently support skip-gram model only.
		"""
		self.n_inputs = vocabulary_size
		self.n_hidden = embedding_dim
		self.n_outputs = self.n_inputs
		
		self.window_size = window_size
		self.batch_size = batch_size
		self.n_skips = n_skips
		self.vocabulary_size = vocabulary_size
		self.neg_k = neg_k
		
		self.embeddings = preprocess.xavier_init(
			(self.n_inputs, self.n_hidden), self.n_inputs, self.n_hidden)
		self.softmax_w = preprocess.xavier_init(
			(self.n_hidden, self.n_outputs), self.n_hidden, self.n_outputs)
		self.softmax_b = np.zeros(self.n_outputs)

		self.previous_d_softmax_w = np.zeros(self.softmax_w.shape)
		self.previous_d_softmax_b = np.zeros(self.softmax_b.shape)
		self.previous_d_embeddings = np.zeros(self.embeddings.shape)

	def negative_samples(self, labels):
		samples = []
		vocab = xrange(self.vocabulary_size)
		while len(samples) < self.neg_k - 1:
			choice = np.random.choice(vocab)
			if choice not in labels:
				samples.append(choice)
		return samples

	def Forward(self, data, labels):
		hidden = np.zeros((data.shape[0], self.n_hidden))
		for i in range(data.shape[0]):
			word = data[i]
			hidden[i] = self.embeddings[word, :]

		# Do negative sampled softmax.
		samples = np.zeros((self.batch_size, self.neg_k), dtype=np.int32)
		samples[:, :-1] =  self.negative_samples(labels)
		samples[:, -1] = labels

		output = np.zeros((self.batch_size, self.neg_k))
		for i in range(data.shape[0]):
			output[i, :] = np.dot(hidden[i], self.softmax_w[:, samples[i]]) + self.softmax_b[samples[i]]
			output[i, :] = self.softmax(output[i])

		cache = {
			'hidden': hidden,
			'output': output,
			'data': data,
			'samples': samples
		}
		return cache

	def Backward(self, d_out, cache):
		hidden = cache['hidden']
		data = cache['data']
		samples = cache['samples']
		d_hidden = np.zeros(hidden.shape)
		d_embeddings = np.zeros(self.embeddings.shape)
		d_softmax_b = np.zeros(self.softmax_b.shape)
		d_softmax_w = np.zeros(self.softmax_w.shape)

		for i in range(data.shape[0]):
			d_hidden[i] = np.dot(d_out[i], self.softmax_w[:, samples[i]].T)
			d_softmax_w[:, samples[i]] += np.dot(hidden[i][:, np.newaxis], d_out[i, np.newaxis])
			d_softmax_b[samples[i]] += d_out[i]

		for i in range(d_hidden.shape[0]):
			d_embeddings[data[i], :] += d_hidden.T[:, i]

		d_cache = {
			'd_softmax_b': d_softmax_b,
			'd_softmax_w': d_softmax_w,
			'd_embeddings': d_embeddings
		}
		return d_cache

	def Train(self, data, max_epochs, learning_rate, momentum):
		gen = BatchGenerator(
			self.batch_size, self.window_size, self.n_skips, data)
		self.learning_rate = learning_rate

		for epoch in xrange(max_epochs):
			batch_input, batch_output = gen.next_batch()
			cache = self.Forward(batch_input, batch_output)

			samples = cache['samples']
			# Compute output gradients.
			d_out, loss = self.compute_output_gradient(
				batch_output, cache['output'])
			# Compute other parameter gradients.
			d_cache = self.Backward(d_out, cache)

			# Apply gradients to parameters.
			d_softmax_w = self.normalize_gradient(d_cache['d_softmax_w'])
			self.previous_d_softmax_w = momentum * self.previous_d_softmax_w - learning_rate * d_softmax_w
			self.softmax_w = self.softmax_w + self.previous_d_softmax_w

			d_softmax_b = self.normalize_gradient(d_cache['d_softmax_b'])
			self.previous_d_softmax_b = momentum * self.previous_d_softmax_b - learning_rate * d_softmax_b
			self.softmax_b = self.softmax_b + self.previous_d_softmax_b

			d_embeddings = self.normalize_gradient(self.embeddings)
			self.previous_d_embeddings = momentum * self.previous_d_embeddings - learning_rate * d_embeddings
			self.embeddings = self.embeddings + self.previous_d_embeddings

			if epoch % 100 == 0:
				print "Epoch: %d, Error: %.6f" % (epoch, loss)

	def softmax(self, z):
		ez = np.exp(z)
		return ez / ez.sum()

	def compute_output_gradient(self, labels, outputs):
		# Xentropy loss function.
		target = np.zeros((labels.shape[0], self.neg_k))
		target[:, -1] = 1
		loss = np.sum(-target * np.log(outputs)) / outputs.shape[0]
		d_out = outputs - target
		return d_out, loss

	def normalize_gradient(self, grad):
		return grad / self.batch_size

	def cosine_similarity(self, word, top=10):
		mult = np.sum(self.embeddings[word] * self.embeddings, 1)
		normalizer = np.sum(self.embeddings[word]**2) * np.sum(self.embeddings**2, 1)
		return np.argsort(mult/normalizer)[-1:-top-1:-1]
