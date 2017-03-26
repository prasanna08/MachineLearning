import batch_generators
import numpy as np
import model
import optimizers
import preprocess
import word2vec

"""Simple paragraph 2 vector implementation."""

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


class Doc2Vec(model.SupervisedModel):
	def __init__(
			self, doc_embedding_dim, word_embedding_dim, n_skips,
			window_size, vocabulary_size, neg_k, word2vec_model=None,
			sample_vocabulary=None):
		"""doc2vec model.
		
		Currently supports PV-DM and PV-DBOW version of doc2vec.

		Args:-
			batch_size: int. Batch size of mini batch training.
			n_docs: int. Number of documents for training.
			doc_embedding_dim: int. Dimension of document embeddings.
			word_embedding_dim: int. Dimension of word embeddings.
			n_skips: int. Number of words used in input to predict next output
				word.
			window_size: int. Size of the word window for word2vec model.
			vocabulary_size: int. Size of vocabulary.
			neg_k: int. Negative sampling size.
			word2vec_model. Word2Vec. A pretrained word2vec model
				(if available).
			sample_vocabulary: list of str. Words from which samples will be
				extracted for negative sampling.
		"""
		self.doc_embedding_dim = doc_embedding_dim
		self.word_embedding_dim = word_embedding_dim
		self.n_skips = n_skips
		self.window_size = window_size
		self.vocabulary_size = vocabulary_size
		self.neg_k = neg_k
		self.is_word2vec_trained = False
		self.gen = None
		self.sample_vocabulary = sample_vocabulary

		self.total_embeddings = preprocess.xavier_init(
			(n_docs, 2 * doc_embedding_dim), n_docs, 2 * doc_embedding_dim)

		# Parameters for pv-dm model.
		self.doc_embeddings = self.total_embeddings[:, :doc_embedding_dim]
		self.softmax_w = preprocess.xavier_init((
			doc_embedding_dim + n_skips * word_embedding_dim,
				vocabulary_size),
			doc_embedding_dim + n_skips * word_embedding_dim, vocabulary_size)
		self.softmax_b = np.zeros(vocabulary_size)

		# Parameters for pv-dbow model.
		self.dbow_embeddings = self.total_embeddings[:, doc_embedding_dim:]
		self.dbow_softmax_w = preprocess.xavier_init(
			(doc_embedding_dim, vocabulary_size), doc_embedding_dim,
			vocabulary_size)
		self.dbow_softmax_b = np.zeros(vocabulary_size)

		self.params = self.get_params()

		if word2vec_model is None:
			self.w2v = word2vec.Word2Vec(
				word_embedding_dim, vocabulary_size, window_size,
				n_skips, neg_k)
			self.word_embeddings = self.w2v.embeddings

		else:
			self.w2v = word2vec_model
			self.word_embeddings = word2vec_model.embeddings
			self.word2id = word2vec_model.word2id
			self.id2word = word2vec_model.id2word
			self.is_word2vec_trained = True

	def _encode_paragraph(self, z):
		"""Given a text paragraph convert it into equivalent numeric paragraph.
		"""
		para = tokenize_paragraphs([z])[0]
		return doc2id(z, self.word2id)

	def softmax(self, z):
		ez = z - z.max()
		ez = np.exp(ez)
		return ez / ez.sum()

	def set_sample_vocabulary(self):
		self.sample_vocabulary = self.w2v.sample_vocabulary

	def negative_samples(self, labels):
		return self.w2v.negative_samples(labels)

	def reset_total_embeddings(self):
		self.total_embeddings[:, :self.doc_embedding_dim] = self.doc_embeddings
		self.total_embeddings[:, self.doc_embedding_dim:] = self.dbow_embeddings

	def train_word2vec(self, data, max_epochs, learning_rate):
		data = reduce(lambda x, y: x+y, data)
		opti = optimizers.Adam(self.w2v, 20*self.n_skips, learning_rate)
		opti.train(data)
		self.word2id = self.w2v.word2id
		self.id2word = self.w2v.id2word
		self.is_word2vec_trained = True

	def forward(self, data, labels):
		# Forward pass for pv-dm model.
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

		# Forward pass for pv-dbow model.
		dbow_hidden = self.dbow_embeddings[data[:, 0], :]
		dbow_output = np.zeros((self.batch_size, self.neg_k))
		for i in range(self.batch_size):
			dbow_output[i] = self.softmax(np.dot(dbow_hidden[i], self.dbow_softmax_w[:, samples[i]]) + self.dbow_softmax_b[samples[i]])

		cache = {
			'data': data,
			'hidden': hidden,
			'dbow_hidden': dbow_hidden,
			'output': output,
			'dbow_output': dbow_output,
			'samples': samples
		}
		return cache

	def backward(self, dout, dout_dbow, cache):
		data = cache['data']
		hidden = cache['hidden']
		dbow_hidden = cache['dbow_hidden']
		samples = cache['samples']

		# Gradient for pv-dm model.
		d_hidden = np.zeros(hidden.shape)
		d_doc_embeddings = np.zeros(self.doc_embeddings.shape)
		d_softmax_w = np.zeros(self.softmax_w.shape)
		d_softmax_b = np.zeros(self.softmax_b.shape)

		# Gradient for pv-dbow model.
		d_dbow_hidden = np.zeros(dbow_hidden.shape)
		d_dbow_embeddings = np.zeros(self.dbow_embeddings.shape)
		d_dbow_softmax_w = np.zeros(self.dbow_softmax_w.shape)
		d_dbow_softmax_b = np.zeros(self.dbow_softmax_b.shape)

		for i in range(self.batch_size):
			# Backward pass for pv-dm model.
			d_hidden[i] = np.dot(dout[i], self.softmax_w[:, samples[i]].T)
			d_softmax_w[:, samples[i]] += np.dot(hidden[i][:, np.newaxis], dout[i, np.newaxis])
			d_softmax_b[samples[i]] += dout[i]
			d_doc_embeddings[data[i, 0], :] += d_hidden[i, :self.doc_embedding_dim]

			# Backward pass for pv-dbow model.
			d_dbow_hidden[i] = np.dot(dout_dbow[i], self.dbow_softmax_w[:, samples[i]].T)
			d_dbow_softmax_w[:, samples[i]] += np.dot(dbow_hidden[i][:, np.newaxis], dout_dbow[i, np.newaxis])
			d_dbow_softmax_b[samples[i]] += dout_dbow[i]
			d_dbow_embeddings[data[i, 0], :] += d_dbow_hidden[i]

		d_cache = {
			'd_doc_embeddings': d_doc_embeddings,
			'd_softmax_w': d_softmax_w,
			'd_softmax_b': d_softmax_b,
			'd_dbow_embeddings': d_dbow_embeddings,
			'd_dbow_softmax_w': d_dbow_softmax_w,
			'd_dbow_softmax_b': d_dbow_softmax_b
		}
		return d_cache

	def compute_loss_and_gradient(self, outputs):
		# Xentropy loss function.
		loss = np.sum(-1 * np.log(outputs)[:, -1])
		dout = outputs.copy()
		dout[:, -1] = outputs[:, -1] - 1
		return loss, dout

	def get_batch_generator(self, batch_size, data, labels=None):
		data = docs2id(tokenize_paragraphs(data))
		self.set_sample_vocabulary()
		return batch_generators.Doc2VecBatchGenerator(
			batch_size, data, self.n_skips)

	def get_params_mapping(self):
		mappings = {
			'doc_embeddings': ['d_doc_embeddings', self.doc_embeddings.shape],
			'softmax_w': ['d_softmax_w', self.softmax_w.shape],
			'softmax_b': ['d_softmax_b', self.softmax_b.shape],
			'dbow_embeddings': ['d_dbow_embeddings', self.dbow_embeddings.shape],
			'dbow_softmax_w': ['d_dbow_softmax_w', self.dbow_softmax_w.shape],
			'dbow_softmax_b': ['d_dbow_softmax_b', self.dbow_softmax_b.shape]
		}
		return mappings

	def get_params(self):
		params = {
			'doc_embeddings': self.doc_embeddings,
			'softmax_w': self.softmax_w,
			'softmax_b': self.softmax_b,
			'dbow_embeddings': self.dbow_embeddings,
			'dbow_softmax_w': self.dbow_softmax_w,
			'dbow_softmax_b': self.dbow_softmax_b
		}
		return params

	def train(self, batch_input, batch_output):
		"""Train document 2 vector model."""
		# If word2vec model is not trained then train that first.
		if (not self.is_word2vec_trained) or train_word2vec:
			self.train_word2vec(data, max_epochs, learning_rate)

		# Normalize embeddings.
		self.doc_embeddings /= np.sqrt((self.doc_embeddings ** 2).sum(axis=1)[:, np.newaxis])
		self.dbow_embeddings /= np.sqrt((self.dbow_embeddings ** 2).sum(axis=1)[:, np.newaxis])

		cache = self.forward(batch_input, batch_output)
		loss, dout = self.compute_loss_and_gradient(cache['output'])
		dbow_loss, dout_dbow = self.compute_loss_and_gradient(cache['dbow_output'])
		d_cache = self.backward(dout, dout_dbow, cache)
		return self.params, d_cache, (loss + dbow_loss) / 2

	def vectorize_paragraphs(
			self, paragraphs, max_epochs, learning_rate, momentum):
		"""Create vector representation of a new paragraph.
		
		This is used at time of testing / inference to make decisions about
		unseen paragraphs.
		"""
		new_embeddings = preprocess.xavier_init(
			(len(paragraphs) + self.doc_embeddings.shape[0], 
				self.doc_embeddings.shape[1]))

		new_dbow_embeddings = preprocess.xavier_init(
			(len(paragraphs) + self.dbow_embeddings.shape[0], 
				self.dbow_embeddings.shape[1]))

		new_embeddings[len(paragraphs):, :] = self.doc_embeddings
		new_dbow_embeddings[len(paragraphs):, :] = self.dbow_embeddings

		original_embeddings = self.doc_embeddings.copy()
		original_dbow_embeddings = self.dbow_embeddings.copy()

		self.doc_embeddings = new_embeddings
		self.dbow_embeddings = new_dbow_embeddings

		previous_d_doc_embeddings = np.zeros(new_embeddings.shape)
		previous_d_dbow_embeddings = np.zeros(new_dbow_embeddings.shape)

		gen = self.get_batch_generator(len(paragraphs), paragraphs)

		for epoch in xrange(max_epochs):
			# Forward pass.
			batch_input, batch_output = gen.next_batch()
			cache = self.forward(batch_input, batch_output)

			# Compute output gradients.
			loss, dout = self.compute_loss_and_gradient(cache['output'])
			dbow_loss, dout_dbow = self.compute_loss_and_gradient(cache['dbow_output'])

			# Backward pass.
			d_cache = self.backward(dout, dout_dbow, cache)

			d_doc_embeddings = d_cache['d_doc_embeddings'] / len(paragraphs)
			previous_d_doc_embeddings = momentum * previous_d_doc_embeddings - learning_rate * d_doc_embeddings
			self.doc_embeddings = self.doc_embeddings + previous_d_doc_embeddings

			d_dbow_embeddings = d_cache['d_dbow_embeddings'] / len(paragraphs)
			previous_d_dbow_embeddings = momentum * previous_d_dbow_embeddings - learning_rate * d_dbow_embeddings
			self.dbow_embeddings = self.dbow_embeddings + previous_d_dbow_embeddings

			# Normalize embeddings.
			self.doc_embeddings /= np.sqrt((self.doc_embeddings ** 2).sum(axis=1)[:, np.newaxis])
			self.dbow_embeddings /= np.sqrt((self.dbow_embeddings ** 2).sum(axis=1)[:, np.newaxis])

		self.doc_embeddings = original_embeddings
		self.dbow_embeddings = original_dbow_embeddings
		self.reset_total_embeddings()
		return new_embeddings[:len(paragraphs), :], new_dbow_embeddings[:len(paragraphs), :]

	def cosine_similarity(
			self, para, max_epochs, learning_rate, momentum, top=3):
		"""Find similarity between given paragraph and some other paragraph.
	
		Args:-
			para: str. Paragraph for which similar paragraphs are to be found.
			learning_rate: float. Learning rate during inference.
			momentum: float. Momentum during inference.
			top: int. Number of top similar paragraphs.
		"""
		try:
			self.id2word
			self.word2id
		except Exception:
			print "Please train doc2vec model with data first."
			return None

		para = self._encode_paragraph(para)
		p_dm_vector, p_dbow_vector = self.vectorize_paragraphs(
			[para], max_epochs, learning_rate, momentum)
		p_vector = np.concatenate([p_dm_vector, p_dbow_vector], axis=1)
		simi = np.sum(p_vector * self.total_embeddings, 1)
		return np.argsort(simi)[-1:-top-1:-1]
