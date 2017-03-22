"""This is word2vec model for high quality representation of words."""
import batch_generators
from collections import Counter
import model
import numpy as np
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


class Word2Vec(model.SupervisedModel):
	def __init__(
			self, embedding_dim, vocabulary_size, window_size, n_skips,
			neg_k=64, sample_vocabulary=None):
		"""Word2Vec class.

		It is assumed that data provided is numerical and contains word_ids rather
		than actual words.

		Currently support skip-gram model only.
		"""
		self.n_inputs = vocabulary_size
		self.n_hidden = embedding_dim
		self.n_outputs = self.n_inputs
		
		self.window_size = window_size
		self.n_skips = n_skips
		self.vocabulary_size = vocabulary_size
		self.neg_k = neg_k
		self.sample_vocabulary = sample_vocabulary
		self.sample_vocabulary_size = int(self.vocabulary_size / 5)
		
		self.embeddings = preprocess.xavier_init((self.n_inputs, self.n_hidden))
		self.softmax_w = preprocess.xavier_init((self.n_hidden, self.n_outputs))
		self.softmax_b = np.zeros(self.n_outputs)

		self.params = self.get_params()

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

	def get_params(self):
		return {
			'embeddings': self.embeddings,
			'softmax_w': self.softmax_w,
			'softmax_b': self.softmax_b
		}

	def softmax(self, z):
		ez = z - z.max()
		ez = np.exp(ez)
		return ez / ez.sum()

	def forward(self, data, labels):
		hidden = self.embeddings[data, :]

		# Do negative sampled softmax.
		samples = np.zeros((data.shape[0], self.neg_k), dtype=np.int32)
		samples[:, :-1] =  self.negative_samples(labels)
		samples[:, -1] = labels

		output = np.zeros((data.shape[0], self.neg_k))
		for i in range(data.shape[0]):
			output[i, :] = self.softmax(np.dot(hidden[i], self.softmax_w[:, samples[i]]) + self.softmax_b[samples[i]])

		cache = {
			'hidden': hidden,
			'output': output,
			'data': data,
			'samples': samples
		}
		return cache

	def backward(self, d_out, cache):
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

	def compute_loss_and_gradient(self, outputs):
		# Xentropy loss function.
		loss = np.sum(-1 * np.log(outputs)[:, -1]) / outputs.shape[0]
		dout = outputs.copy()
		dout[:, -1] = outputs[:, -1] - 1
		return loss, dout

	def get_batch_generator(self, batch_size, data, labels=None):
		self.word2id, self.id2word, word_counts, data = word2id(
			data, self.vocabulary_size)
		self.set_sample_vocabulary(word_counts, self.word2id)
		return batch_generators.Word2VecBatchGenerator(
			batch_size, data, self.window_size, self.n_skips)

	def get_params_mapping(self):
		mapper = {
			'embeddings': ['d_embeddings', self.embeddings.shape],
			'softmax_w': ['d_softmax_w', self.softmax_w.shape],
			'softmax_b': ['d_softmax_b', self.softmax_b.shape]
		}
		return mapper

	def train(self, batch_input, batch_output):
		# Normalize embeddings.
		self.embeddings /= np.sqrt((self.embeddings ** 2).sum(axis=1)[:, np.newaxis])
		cache = self.forward(batch_input, batch_output)
		loss, dout = self.compute_loss_and_gradient(cache['output'])
		d_cache = self.backward(dout, cache)
		return self.params, d_cache, loss

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
