import numpy as np

"""This file contains some functions related to preprocessing."""

def onehot(output_labels):
	return np.squeeze((np.unique(output_labels) == output_labels[:, None]).astype(np.float32))

def xavier_init(shape, fan_in, fan_out):
	dev = np.sqrt(6.0 / (fan_in + fan_out))
	return np.random.uniform(-dev, dev, shape)

def xavier_init_normal(shape, fan_in, fan_out):
	dev = np.sqrt(3.0 / (fan_in + fan_out))
	return np.random.normal(size=shape, scale=dev)