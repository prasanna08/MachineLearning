import abc
import numpy as np

"""BatchGenerator classes for various models."""

class BatchGenerator(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, batch_size, data, labels):
		self.batch_size = batch_size
		self.data = data
		self.labels = labels

	@abc.abstractmethod
	def next_batch(self):
		"""This function will return nextbatch."""
		return


class EncoderBatchGenerator(BatchGenerator):
	def __init__(self, batch_size, data):
		super(EncoderBatchGenerator, self).__init__(batch_size, data, None)
		self._cursor = 0
		self.data_length = data.shape[0]

	def next_batch(self):
		old_cursor = self._cursor
		self._cursor = (self._cursor + 1) % (self.data_length - self.batch_size)
		return self.data[old_cursor: old_cursor+self.batch_size, :]


class FFNBatchGenerator(BatchGenerator):
	def __init__(self, batch_size, data, labels):
		super(FFNBatchGenerator, self).__init__(batch_size, data, labels)
		self._cursor = 0
		self.data_length = data.shape[0]

	def next_batch(self):
		batch_input = self.data[self._cursor: self._cursor+self.batch_size]
		batch_outut = self.labels[self._cursor: self._cursor+self.batch_size]
		self._cursor = (self._cursor + 1) % (self.data_length - self.batch_size)
		return batch_input, batch_outut


class Word2VecBatchGenerator(BatchGenerator):
	def __init__(
			self, batch_size, data, window_size, n_skips, skip_gram=True):
		super(Word2VecBatchGenerator, self).__init__(batch_size, data, None)
		self.window_size = window_size
		self.n_skips = n_skips
		self._cursor = 0
		self.skip_gram = skip_gram

		# batch_size should be multiple of n_skips because each target word
		# is present in batch for n_skips times.
		assert batch_size % n_skips == 0

	def next_batch(self):
		if self.skip_gram:
			return self.next_batch_skip()
		else:
			return self.next_batch_cbow()

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

		for i in range(self.batch_size // self.n_skips):
			target = words[self.window_size]

			order = range(len(words))
			order.remove(self.window_size)
			np.random.shuffle(order)
			
			labels = [words[x] for x in order[:self.n_skips]]
			for k in range(self.n_skips):
				batch_input[i*self.n_skips + k] = target
				batch_output[i*self.n_skips + k] = labels[k]

			self._cursor = (self._cursor + 1) % (len(self.data) - (2*self.window_size))
			words = words[1:]
			words.append(self.data[self._cursor])

		return batch_input, batch_output


class Doc2VecBatchGenerator(BatchGenerator):
	def __init__(self, batch_size, data, n_skips):
		super(Doc2VecBatchGenerator, self).__init__(batch_size, data, labels)
		self.n_skips = n_skips
		self._cursor = [0 for i in xrange(len(data))]

	def next_batch(self):
		doc_indices = np.random.choice(len(self.data), self.batch_size)
		batch = np.ndarray(shape=(self.batch_size, self.n_skips + 2), dtype=np.int32)
		k = 0
		for i in doc_indices:
			batch[k, 0] = i
			cursor = self._cursor[i]
			for j in xrange(self.n_skips + 1):
				try:
					batch[k, j+1] = self.data[i][cursor]
					cursor = (cursor + 1)
				except Exception as e:
					print cursor
					print self.data[i]
			self._cursor[i] = (self._cursor[i] + 1) % (len(self.data[i]) - self.n_skips)
			k += 1
		return batch[:, :-1], batch[:, -1]
