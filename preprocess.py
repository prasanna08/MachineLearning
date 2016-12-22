import numpy as np

"""This file contains some functions related to preprocessing."""

def onehot(output_labels):
	return np.squeeze((np.unique(output_labels) == output_labels[:, None]).astype(np.float32))
