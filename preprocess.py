import numpy as np

"""This file contains some functions related to preprocessing."""

def get_labels(output_labels):
	return (np.unique(output_labels) == output_labels[:, None]).astype(np.float32)
