import numpy as np
from im2col.im2col_cython import col2im_cython, im2col_cython

"""Useful functions for convolution operations.

Functions are obtained from assignment 2 of CS231n course of stanford.
"""

def conv_forward(x, w, b, pad, stride):
	N, H, W, C = x.shape
	filter_height, filter_width, _, num_filters = w.shape

	# Check dimensions
	assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
	assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

	# Create output
	out_height = (H + 2 * pad - filter_height) / stride + 1
	out_width = (W + 2 * pad - filter_width) / stride + 1

	x_cols = im2col_cython(x, filter_height, filter_width, pad, stride)
	res = w.transpose(3, 2, 0, 1).reshape((num_filters, -1)).dot(x_cols) + b.reshape(-1, 1)

	out = res.reshape(num_filters, out_height, out_width, N)
	out = out.transpose(3, 1, 2, 0)

	cache = (x, w, b, pad, stride, x_cols)
	return out, cache

def conv_backward(dout, cache):
	x, w, b, pad, stride, x_cols = cache

	db = np.sum(dout, axis=(0, 1, 2))
	filter_height, filter_width, _, num_filters = w.shape
	dout_reshaped = dout.transpose(3, 1, 2, 0).reshape(num_filters, -1)
	dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

	dx_cols = w.transpose(3, 2, 0, 1).reshape(num_filters, -1).T.dot(dout_reshaped)
	# dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
	dx = col2im_cython(
		dx_cols, x.shape[0], x.shape[3], x.shape[1], x.shape[2],
		filter_height, filter_width, pad, stride)

	return dx, dw, db

def max_pool_forward(x, pool_height, pool_width, stride):
	N, H, W, C = x.shape

	same_size = pool_height == pool_width == stride
	tiles = H % pool_height == 0 and W % pool_width == 0
	if same_size and tiles:
		out, reshape_cache = max_pool_forward_reshape(x, pool_param)
		cache = ('reshape', reshape_cache)
	else:
		out, im2col_cache = max_pool_forward_im2col(x, pool_param)
		cache = ('im2col', im2col_cache)
	return out, cache

def max_pool_forward_reshape(x, pool_height, pool_width, stride):
	N, H, W, C = x.shape
	assert pool_height == pool_width == stride, 'Invalid pool params'
	assert H % pool_height == 0
	assert W % pool_height == 0
	x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
	                     W / pool_width, pool_width)
	out = x_reshaped.max(axis=3).max(axis=4).transpose(0, 2, 3, 1)

	cache = (x, x_reshaped, out)
	return out, cache

def max_pool_forward_im2col(x, pool_height, pool_width, stride):
	N, H, W, C = x.shape
	assert (H - pool_height) % stride == 0, 'Invalid height'
	assert (W - pool_width) % stride == 0, 'Invalid width'

	out_height = (H - pool_height) / stride + 1
	out_width = (W - pool_width) / stride + 1

	x_split = x.reshape(N * C, 1, H, W)
	x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
	x_cols_argmax = np.argmax(x_cols, axis=0)
	x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
	out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 0, 1, 3)

	cache = (x, x_cols, x_cols_argmax, pool_param)
	return out, cache

def max_pool_backward(dout, cache):
	method, real_cache = cache
	if method == 'reshape':
		return max_pool_backward_reshape(dout, real_cache)
	elif method == 'im2col':
		return max_pool_backward_im2col(dout, real_cache)
	else:
		raise ValueError('Unrecognized method "%s"' % method)

def max_pool_backward_reshape(dout, cache):
	x, x_reshaped, out = cache
	dx_reshaped = np.zeros_like(x_reshaped)
	out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
	mask = (x_reshaped == out_newaxis)
	dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
	dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
	dx_reshaped[mask] = dout_broadcast[mask]
	dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
	dx = dx_reshaped.reshape(x.shape)
	return dx

def max_pool_backward_im2col(dout, cache):
	x, x_cols, x_cols_argmax, pool_param = cache
	N, H, W, C = x.shape
	pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
	stride = pool_param['stride']

	dout_reshaped = dout.transpose(1, 2, 0, 3).flatten()
	dx_cols = np.zeros_like(x_cols)
	dx_cols[x_cols_argmax, np.arange(dx_cols.shape[3])] = dout_reshaped
	dx = col2im_cython(
		dx_cols, N, C, H, W, pool_height, pool_width, padding=0, stride=stride)
	dx = dx.reshape(x.shape)

	return dx
