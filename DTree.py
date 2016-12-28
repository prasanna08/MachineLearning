import numpy as np

class DTree(object):
	"""This is basic implementation of decision tree.

	As of now it only works on data that has descrete feature values.
	If any feature of input data has continuos value then this will not work.
	"""
	def __init__(self):
		pass

	def make_tree(self, data, classes, feature_names=None):
		num_data = data.shape[0]
		num_features = data.shape[1]
		new_classes = np.unique(classes)
		frequency = np.zeros(len(new_classes), dtype=float)

		total_entropy = 0
		total_gini = 0
		classIndex = 0
		for aclass in new_classes:
			frequency[classIndex] = np.sum(classes == aclass)
			total_entropy += self.calc_entropy(frequency[classIndex] / num_data)
			total_gini += (frequency[classIndex] / num_data ** 2)
			classIndex += 1

		total_gini = 1 - total_gini
		default = int(new_classes[frequency.argmax()])

		if num_data == 0 or num_features == 0:
			return default
		elif len(new_classes) == 1:
			return int(new_classes[0])
		else:
			gain = np.zeros(num_features)
			ggain = np.zeros(num_features)
			for feature in range(num_features):
				g, gg = self.calc_info_gain(data, classes, feature)
				gain[feature] = total_entropy - g
				ggain[feature] = total_gini - gg

			best_feature = gain.argmax()
			f_values = np.unique(data[:, best_feature])
			tree = {best_feature: {}}
			for value in f_values:
				points = np.where(data[:, best_feature] == value)[0]
				new_data = np.concatenate([data[points, :best_feature], data[points, best_feature+1:]], axis=1)
				new_data_classes = classes[points]
				#new_feature_names = feature_names[:best_feature]
				#new_feature_names.extend(feature_names[best_feature+1:])
				subtree = self.make_tree(new_data, new_data_classes)
				tree[best_feature][value] = subtree

			return tree

	def classify(self, tree, data):
		if type(tree) == type({}):
			feature = tree.keys()[0]
			value = data[feature]
			cls = self.classify(tree[feature][value], data)
			return cls
		else:
			return tree

	def calc_entropy(self, p):
		if p != 0:
			return -1 * p * np.log2(p)
		else:
			return 0

	def calc_info_gain(self, data, classes, feature, continuous=False):
		if not continuous:
			f_values = np.unique(data[:, feature])
			f_classes = np.unique(classes)
			entropy = np.zeros((len(f_values), len(f_classes)), dtype=float)

			for valueIndex in range(len(f_values)):
				points = np.where(data[:, feature] == f_values[valueIndex])[0]
				entropy[valueIndex, :] = np.sum(classes[points] == f_classes, axis=0)
		else:
			mean = np.mean(data[:, feature], axis=0)
			f_classes = np.unique(classes)
			entropy = np.zeros((2, len(f_classes)), dtype=float)
			points = np.where(data[:, feature] <= mean)
			entropy[0, :] = np.sum(classes[points] == f_classes, axis=0)
			points = np.where(data[:, feature] > mean)
			entropy[1, :] = np.sum(classes[points] == f_classes, axis=0)

		points_per_value = entropy.sum(axis=1)
		entropy /= points_per_value.reshape(len(points_per_value), 1)
		logp = entropy.copy()
		logp[logp == 0] = 1
		logp = np.log2(logp)
		ggain = np.sum(entropy ** 2, axis=1)
		entropy = (-1 * entropy * logp)
		gain = np.sum(entropy.sum(axis=1) * points_per_value / data.shape[0])
		ggain = np.sum(ggain * points_per_value / data.shape[0])
		return gain, 1 - ggain
