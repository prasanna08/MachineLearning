import numpy as np

class KNN(object):
	def __init__(
			self, inputs, outputs, k, outtype='classification',
			distance_metric='LK', k_metric=2,
			linear_output_func='quadratic', lam=2):
		"""Class that implements K Nearest Neighbours learning algorithm.
		
		Args:
		  	inputs: numpy array of inputs.
		  	outputs: numpy array of outputs.
		  	k: number of nearest neighbours to consider.
		  	outtype: whether output is linear or classification. Possible
		  		choices are ['classification', 'regression'].
		  	distance_metric: function to use for calculating relative distance
		  	  	between 2 points. If custom function is used then set this to
		  	  	that custom function. Make sure custom function has only 2
		  	  	inputs, namely: all_inputs (all input points), input (point
		  	  	whose distance is to be measured).
			k_metric: if distance metric is 'LK' then k_metric is value of K
				for e.g. L1 distance(Manhatten), L2 distance(Euclidean).
			linear_output_func: type linear output function to use. This is used
				only when outtype == 'regression'. Possible choices are
				['quadratic', 'tricubic']
			lam: value of lambda for calculating linear output function.
		"""
		self.inputs = np.array(inputs)
		if self.inputs.ndim == 1:
			self.inputs = self.inputs.reshape(self.inputs.shape[0], 1)

		self.outputs = np.array(outputs)
		if self.outputs.ndim == 1:
			self.outputs = self.outputs.reshape(self.outputs.shape[0], 1)

		self.outfuncs = {
			'classification': self.get_class_output
		}
		self.k = k
		self.outtype = outtype
		self.distance_metric = distance_metric
		self.k_metric = k_metric
		self.lam = lam
		self.linear_output_func = linear_output_func

	def calculate_distances(self, data_input):
		distance = np.absolute((self.inputs - data_input))
		if self.distance_metric == 'LK':
			return np.sum(distance**self.k_metric, axis=1)
		else:
			return distance_metric(self.inputs, data_input)

	def get_class_output(self, ndataeighbours):
		classes = np.unique(self.outputs[neighbours])
		if len(classes) == 1:
			return classes[0]
		else:
			max_count = -1
			for c in classes:
				cur_count = (self.outputs[neighbours] == c).sum()
				if cur_count > max_count:
					max_count = cur_count
					max_class = c
			return max_class

	def get_regression_output(self, data_input, neighbours):
		distances = np.absolute(self.inputs[neighbours] - data_input)
		if self.linear_output_func == 'quadratic':
			neighbour_vals = 0.75*(1 - (np.sum(distances, axis=1)/self.lam)**2)
		elif self.linear_output_func == 'tricubic':
			neighbour_vals = (1 - (np.sum(distances, axis=1)/self.lam)**3)**3
		neighbour_vals[neighbour_vals<0] = 0
		return np.sum(neighbour_vals)/neighbour_vals.shape[0]

	def find_nearest(self, data_input):
		distances = self.calculate_distances(data_input)
		indices = distances.argsort()
		indices = indices[:self.k]
		return indices

	def get_outputs(self, inputs):
		outputs = np.zeros((inputs.shape[0], 1))
		for i in xrange(inputs.shape[0]):
			indices = self.find_nearest(inputs[i])
			if self.outtype == 'classification':
				outputs[i] = self.get_class_output(indices)
			elif self.outtype == 'regression':
				outputs[i] = self.get_regression_output(inputs[i], neighbours)
		return outputs
