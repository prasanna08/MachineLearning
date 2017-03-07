"""Simple paragraph 2 vector implementation.
Work In Progress module.
"""

import numpy as np

class BatchGenerator(object):
	def __init__(self, docs, batch_size, n_skips):
		self.docs = docs
		self.batch_size = batch_size
		self.n_skips = n_skips
		seld._cursor = [0 for i in xrange(len(docs))]

	def next_batch(self):
		doc_indices = np.random.choice(len(self.docs), self.batch_size)
		batch = np.ndarray(shape=(batch_size, n_skips + 1))
		k = 0
		for i in doc_indices:
			batch[k, 0] = i
			for j in xrange(self.n_skips):
				batch[k, j+1] = self.docs[i][self._cursor[i]]
				self._cursor[i] = (self._cursor[i] + 1)
			if self._cursor[i] > len(self.docs[i]) - self.n_skips:
				self._cursor[i] = 0
			k += 1
		return batch

class Doc2Vec(object):
	def __init__(self):
		pass
