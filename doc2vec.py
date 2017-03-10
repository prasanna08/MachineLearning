"""Simple paragraph 2 vector implementation.
Work In Progress module.
"""

import numpy as np
import word2vec
import preprocess

def read_file(self, fname):
	with open(fname, 'r') as f:
		data = f.read()
		paras = data.split('\n\n')
		docs = [para.split() for para in paras]
		return docs

def tokenize_paragraphs(paragraphs):
	tokenized = []
	for para in paragraphs:
		z = para.split('\n')
		lines = []
		for line in z:
			em = line.split('.')
			lines.extend(em)

		words = []
		for l in lines:
			em = l.split(' ')
			words.extend(em)

		tokenized.append(filter(lambda w: w != '', words))
	return tokenized

def doc2id(doc, word2id):
	indexed_doc = []
	for word in doc:
		if word in word2id:
			indexed_doc.append(word2id[word])
		else:
			indexed_doc.append(word2id['UNK'])
	return indexed_doc

def docs2id(docs, word2id):
	return [doc2id(doc, word2id) for doc in docs]

def id2doc(doc, id2word):
	return ' '.join(id2word[word] for word in doc)

def ids2doc(docs, id2word):
	return [id2doc(doc, id2word) for doc in docs]

class BatchGenerator(object):
	def __init__(self, docs, batch_size, n_skips):
		self.docs = docs
		self.batch_size = batch_size
		self.n_skips = n_skips
		self._cursor = [0 for i in xrange(len(docs))]

	def next_batch(self):
		doc_indices = np.random.choice(len(self.docs), self.batch_size)
		batch = np.ndarray(shape=(self.batch_size, self.n_skips + 2), dtype=np.int32)
		k = 0
		for i in doc_indices:
			batch[k, 0] = i
			for j in xrange(self.n_skips + 1):
				batch[k, j+1] = self.docs[i][self._cursor[i]]
				self._cursor[i] = (self._cursor[i] + 1)
			if self._cursor[i] > len(self.docs[i]) - self.n_skips:
				self._cursor[i] = 0
			k += 1
		return batch[:, :-1], batch[:, -1]


class Doc2Vec(object):
	def __init__(
			self, batch_size, n_docs, doc_embedding_dim, word_embedding_dim, n_skips,
			window_size, vocabulary_size, neg_k, word2vec_model=None,
			sample_vocabulary=None):
		"""doc2vec model.

		TODO: Implement DOM version of doc2vec.
		TODO: Implement function for inference of new paragraph.
		"""

		self.batch_size = batch_size
		self.n_docs = n_docs
		self.doc_embedding_dim = doc_embedding_dim
		self.word_embedding_dim = word_embedding_dim
		self.n_skips = n_skips
		self.window_size = window_size
		self.vocabulary_size = vocabulary_size
		self.neg_k = neg_k
		self.is_word2vec_trained = False
		self.gen = None
		self.sample_vocabulary = sample_vocabulary

		self.doc_embeddings = preprocess.xavier_init(
			(n_docs, doc_embedding_dim), n_docs, doc_embedding_dim)
		self.softmax_w = preprocess.xavier_init((
			doc_embedding_dim + n_skips * word_embedding_dim,
				vocabulary_size),
			doc_embedding_dim + n_skips * word_embedding_dim, vocabulary_size)
		self.softmax_b = np.zeros(vocabulary_size)

		self.previous_d_softmax_w = np.zeros(self.softmax_w.shape)
		self.previous_d_softmax_b = np.zeros(self.softmax_b.shape)
		self.previous_d_doc_embeddings = np.zeros(self.doc_embeddings.shape)

		if word2vec_model is None:
			self.w2v = word2vec.Word2Vec(
				batch_size, window_size, n_skips, vocabulary_size,
				word_embedding_dim, neg_k)
			self.word_embeddings = self.w2v.embeddings

		else:
			self.w2v = word2vec_model
			self.word_embeddings = word2vec_model.embeddings
			self.word2id = word2vec_model.word2id
			self.id2word = word2vec.id2word
			self.is_word2vec_trained = True

	def _encode_paragraph(self, z):
		"""Given a text paragraph convert it into equivalent numeric paragraph.
		"""
		para = tokenize_paragraphs([z])[0]
		enc_para = [self.word2id[word] for word in para]
		return enc_para	

	def softmax(self, z):
		ez = z - z.max()
		ez = np.exp(ez)
		return ez / ez.sum()

	def set_sample_vocabulary(self):
		self.sample_vocabulary = self.w2v.sample_vocabulary

	def negative_samples(self, labels):
		return self.w2v.negative_samples(labels)

	def Forward(self, data, labels):
		hidden = np.zeros((data.shape[0], self.softmax_w.shape[0]))
		hidden[:, :self.doc_embedding_dim] = self.doc_embeddings[data[:, 0], :]
		
		for i in range(self.n_skips):
			hidden[:, self.doc_embedding_dim + (i * self.word_embedding_dim) : self.doc_embedding_dim + ((i+1) * self.word_embedding_dim)] = self.word_embeddings[data[:, i+1], :]

		samples = np.zeros((data.shape[0], self.neg_k), dtype=np.int32)
		samples[:, :-1] = self.negative_samples(labels)
		samples[:, -1] = labels

		output = np.zeros((self.batch_size, self.neg_k))
		for i in range(self.batch_size):
			output[i] = self.softmax(np.dot(hidden[i], self.softmax_w[:, samples[i]]) + self.softmax_b[samples[i]])

		cache = {
			'data': data,
			'hidden': hidden,
			'output': output,
			'samples': samples
		}
		return cache

	def Backward(self, dout, cache):
		data = cache['data']
		hidden = cache['hidden']
		samples = cache['samples']
		d_hidden = np.zeros(hidden.shape)
		d_doc_embeddings = np.zeros(self.doc_embeddings.shape)
		d_softmax_w = np.zeros(self.softmax_w.shape)
		d_softmax_b = np.zeros(self.softmax_b.shape)

		for i in range(self.batch_size):
			d_hidden[i] = np.dot(dout[i], self.softmax_w[:, samples[i]].T)
			d_softmax_w[:, samples[i]] += np.dot(hidden[i][:, np.newaxis], dout[i, np.newaxis])
			d_softmax_b[samples[i]] += dout[i]
			d_doc_embeddings[data[i, 0], :] += d_hidden[i, :self.doc_embedding_dim]

		d_cache = {
			'd_doc_embeddings': d_doc_embeddings,
			'd_softmax_w': d_softmax_w,
			'd_softmax_b': d_softmax_b
		}
		return d_cache

	def compute_output_gradient(self, outputs):
		# Xentropy loss function.
		loss = np.sum(-1 * np.log(outputs)[:, -1]) / outputs.shape[0]
		dout = outputs.copy()
		dout[:, -1] = outputs[:, -1] - 1
		return dout, loss

	def normalize_gradient(self, grad):
		return grad / self.batch_size

	def Train_word2vec(self, data, max_epochs, learning_rate, momentum):
		data = reduce(lambda x, y: x+y, data)
		self.w2v.Train(data, max_epochs, learning_rate, momentum)
		self.word2id = self.w2v.word2id
		self.id2word = self.w2v.id2word
		self.is_word2vec_trained = True

	def Train(
			self, data, max_epochs, learning_rate, momentum,
			train_word2vec=False):
		"""Train document 2 vector model.

		Currently supports DSM version of doc2vec.
		TODO: Implement DOM version of doc2vec.
		"""
		# If word2vec model is not trained then train that first.
		if (not self.is_word2vec_trained) or train_word2vec:
			self.Train_word2vec(data, max_epochs, learning_rate, momentum)

		# If batch generator is not present create one.
		if self.gen is None:
			data = docs2id(data, self.word2id)
			self.gen = BatchGenerator(data, self.batch_size, self.n_skips)

		if self.sample_vocabulary is None:
			self.set_sample_vocabulary()

		# Use average loss to get a good estimation.
		average_loss = 0
		for epoch in xrange(max_epochs):
			batch_input, batch_output = self.gen.next_batch()
			cache = self.Forward(batch_input, batch_output)

			samples = cache['samples']
			# Compute output gradients.
			dout, loss = self.compute_output_gradient(cache['output'])
			# Compute other parameter gradients.
			d_cache = self.Backward(dout, cache)

			# Apply gradients to parameters.
			d_softmax_w = self.normalize_gradient(d_cache['d_softmax_w'])
			self.previous_d_softmax_w = momentum * self.previous_d_softmax_w - learning_rate * d_softmax_w
			self.softmax_w = self.softmax_w + self.previous_d_softmax_w

			d_softmax_b = self.normalize_gradient(d_cache['d_softmax_b'])
			self.previous_d_softmax_b = momentum * self.previous_d_softmax_b - learning_rate * d_softmax_b
			self.softmax_b = self.softmax_b + self.previous_d_softmax_b

			d_doc_embeddings = self.normalize_gradient(d_cache['d_doc_embeddings'])
			self.previous_d_doc_embeddings = momentum * self.previous_d_doc_embeddings - learning_rate * d_doc_embeddings
			self.doc_embeddings = self.doc_embeddings + self.previous_d_doc_embeddings

			# Normalize embeddings.
			self.doc_embeddings /= np.sqrt((self.doc_embeddings ** 2).sum(axis=1)[:, np.newaxis])

			average_loss += loss
			if epoch % 100 == 0:
				if epoch > 0:
					loss = average_loss / 100
				print "Epoch: %d, Error: %.6f" % (epoch, loss)
				average_loss = 0

	def vectorize_paragraph(self, para):
		"""Create vector representation of a new paragraph.
		
		This is used at time of testing / inference to make decisions about
		unseen paragraphs.
		"""
		pass

	def cosine_similarity(self, para, top=10):
		"""Find similarity between given paragraph and some other paragraph."""
		para = self._encode_paragraph(para)
		p_vector = self.vectorize_paragraph(para)
		simi = np.sum(p_vector * self.doc_embeddings, 1)
		return ids2doc(np.argsort(simi)[-1:-top-1:-1], self.id2word)