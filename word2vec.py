"""This is word2vec model for high quality representation of words."""
import numpy as np
from collections import Counter
import tensorflow as tf
import preprocess

def read_file(file_name):
	f = open(file_name, 'r')
	words = f.read().split()
	return words

def word2id(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocabulary_size - 1))
	word2id = dict()
	
	for word, _ in count:
		word2id[word] = len(word2id)

	unk_count = 0
	indexed_data = []
	for word in words:
		if word in word2id:
			index = word2id[word]
		else:
			index = 0
			unk_count += 1
		indexed_data.append(index)

	count[0][1] = unk_count
	id2word = dict(zip(word2id.values(), word2id.keys()))
	return word2id, id2word, count, indexed_data


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

			self._cursor = (self._cursor + 1) % (len(self.data) - self.window_size)
			words.pop(0)
			words.append(self.data[self._cursor])

		return batch_input, batch_output


class Word2Vec(object):
	def __init__(
			self, batch_size, window_size, n_skips, vocabulary_size,
			embedding_dim, neg_k=64, sample_vocabulary=None):
		"""Word2Vec class.

		It is assumed that data provided is numerical and contains word_ids rather
		than actual words.

		Currently support skip-gram model only.
		"""
		self.n_inputs = vocabulary_size
		self.n_hidden = embedding_dim
		self.n_outputs = self.n_inputs
		
		self.window_size = window_size
		self.batch_size = batch_size
		self.n_skips = n_skips
		self.vocabulary_size = vocabulary_size
		self.neg_k = neg_k
		self.sample_vocabulary = sample_vocabulary
		self.sample_vocabulary_size = int(self.vocabulary_size / 5)
		self.gen = None
		
		self.embeddings = preprocess.xavier_init(
			(self.n_inputs, self.n_hidden), self.n_inputs, self.n_hidden)
		self.softmax_w = preprocess.xavier_init(
			(self.n_hidden, self.n_outputs), self.n_hidden, self.n_outputs)
		self.softmax_b = np.zeros(self.n_outputs)

		self.previous_d_softmax_w = np.zeros(self.softmax_w.shape)
		self.previous_d_softmax_b = np.zeros(self.softmax_b.shape)
		self.previous_d_embeddings = np.zeros(self.embeddings.shape)

	def negative_samples(self, labels):
		vocab = xrange(self.sample_vocabulary_size)
		sample_indices = np.random.choice(vocab, self.neg_k - 1)
		samples = []
		for ind in sample_indices:
			if self.sample_vocabulary[ind] in labels:
				choice  = np.random.choice(self.vocabulary_size)
				while choice in labels:
					choice  = np.random.choice(self.vocabulary_size)
				samples.append(choice)
			else:
				samples.append(self.sample_vocabulary[ind])
		return samples

	def set_sample_vocabulary(self, word_counts, word2id):
		self.sample_vocabulary = np.array([word2id[w] for w, c in word_counts[:self.sample_vocabulary_size]])

	def Forward(self, data, labels):
		hidden = self.embeddings[data, :]

		# Do negative sampled softmax.
		samples = np.zeros((self.batch_size, self.neg_k), dtype=np.int32)
		samples[:, :-1] =  self.negative_samples(labels)
		samples[:, -1] = labels

		output = np.zeros((self.batch_size, self.neg_k))
		for i in range(data.shape[0]):
			output[i, :] = self.softmax(np.dot(hidden[i], self.softmax_w[:, samples[i]]) + self.softmax_b[samples[i]])

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
			d_embeddings[data[i], :] += d_hidden[i]

		d_cache = {
			'd_softmax_b': d_softmax_b,
			'd_softmax_w': d_softmax_w,
			'd_embeddings': d_embeddings
		}
		return d_cache

	def Train(self, data, max_epochs, learning_rate, momentum):
		# If batch generator is not present create one.
		if self.gen is None:
			self.word2id, self.id2word, word_counts, data = word2id(
				data, self.vocabulary_size)
			self.gen = BatchGenerator(
				self.batch_size, self.window_size, self.n_skips, data)

		if self.sample_vocabulary is None:
			self.set_sample_vocabulary(word_counts, self.word2id)
			del word_counts

		# Use average loss to get a good estimation.
		average_loss = 0
		for epoch in xrange(max_epochs):
			batch_input, batch_output = self.gen.next_batch()
			cache = self.Forward(batch_input, batch_output)

			samples = cache['samples']
			# Compute output gradients.
			d_out, loss = self.compute_output_gradient(cache['output'])
			# Compute other parameter gradients.
			d_cache = self.Backward(d_out, cache)

			# Apply gradients to parameters.
			d_softmax_w = self.normalize_gradient(d_cache['d_softmax_w'])
			self.previous_d_softmax_w = momentum * self.previous_d_softmax_w - learning_rate * d_softmax_w
			self.softmax_w = self.softmax_w + self.previous_d_softmax_w

			d_softmax_b = self.normalize_gradient(d_cache['d_softmax_b'])
			self.previous_d_softmax_b = momentum * self.previous_d_softmax_b - learning_rate * d_softmax_b
			self.softmax_b = self.softmax_b + self.previous_d_softmax_b

			d_embeddings = self.normalize_gradient(d_cache['d_embeddings'])
			self.previous_d_embeddings = momentum * self.previous_d_embeddings - learning_rate * d_embeddings
			self.embeddings = self.embeddings + self.previous_d_embeddings

			# Normalize embeddings.
			self.embeddings /= np.sqrt((self.embeddings ** 2).sum(axis=1)[:, np.newaxis])

			average_loss += loss
			if epoch % 100 == 0:
				if epoch > 0:
					loss = average_loss / 100
				print "Epoch: %d, Error: %.6f" % (epoch, loss)
				average_loss = 0

	def softmax(self, z):
		ez = z - z.max()
		ez = np.exp(ez)
		return ez / ez.sum()

	def compute_output_gradient(self, outputs):
		# Xentropy loss function.
		loss = np.sum(-1 * np.log(outputs)[:, -1]) / outputs.shape[0]
		d_out = outputs.copy()
		d_out[:, -1] = outputs[:, -1] - 1
		return d_out, loss	

	def normalize_gradient(self, grad):
		return grad / self.batch_size

	def cosine_similarity(self, word, top=10):
		try:
			self.id2word
			self.word2id
		except Exception:
			print "Please train word2vec model with data first."
			return None

		word_id = self.word2id[word]
		simi = np.sum(self.embeddings[word_id] * self.embeddings, 1)
		sim_ids = np.argsort(simi)[-1:-top-1:-1]
		return [self.id2word[i] for i in sim_ids]
