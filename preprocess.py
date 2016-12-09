import numpy as np

"""This file contains some functions related to preprocessing."""

def get_output_array_from_labels(output_labels, labels_encoding=None):
	labels = np.unique(output_labels)
	labels = labels.reshape(len(labels), 1)
	outputs = np.zeros((output_labels.shape[0], labels.shape[0]))

	if not labels_encoding:
		labels_encoding = np.concatenate(
			[labels, np.eye(labels.shape[0])], axis=1)

	for enc in labels_encoding:
		indices = np.where(output_labels == enc[0])
		outputs[indices[0]] = enc[1:]
	return outputs

labeling = get_output_array_from_labels