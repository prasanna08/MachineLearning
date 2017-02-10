"""Neural network built using tensorflow for MNIST data set."""
import tensorflow as tf
import numpy as np
import cPickle as pickle
import preprocess

def calculate_accuracy(prediction, target):
	return (100.0 * np.sum(prediction.argmax(axis=1) == target.argmax(axis=1))
			/ prediction.shape[0])


data = pickle.load(open('MNIST_dataset.pickle', 'rb'))
train_data = data['train_data']
train_labels = preprocess.onehot(data['train_labels'])
valid_data = data['valid_data']
valid_labels = preprocess.onehot(data['valid_labels'])
test_data = data['test_data']
test_labels = preprocess.onehot(data['test_labels'])
train_data = np.reshape(train_data, [-1, 28, 28, 1])
valid_data = np.reshape(valid_data, [-1, 28, 28, 1])
test_data = np.reshape(test_data, [-1, 28, 28, 1])

batch_size = 128
height = 28
width = 28
filter_size = 5
channel_in = 1
conv1 = 4
conv2 = 16
fc_1 = 512
fc_2 = 512
n_outputs = train_labels.shape[1]
graph = tf.Graph()

# Build NN.
with graph.as_default():
	tf_train_data = tf.placeholder(
		tf.float32, shape=(batch_size, height, width, channel_in))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_outputs))
	tf_valid_data = tf.constant(valid_data, dtype=tf.float32)
	tf_test_data = tf.constant(test_data, dtype=tf.float32)

	w_conv_1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, channel_in, conv1]))
	b_conv_1 = tf.Variable(tf.zeros([conv1]))
	
	w_conv_2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, conv1, conv2]))
	b_conv_2 = tf.Variable(tf.truncated_normal([conv2]))

	w_fc_1 = tf.Variable(tf.truncated_normal([7 * 7 * conv2, fc_1]))
	b_fc_1 = tf.Variable(tf.zeros([fc_1]))

	w_fc_2 = tf.Variable(tf.truncated_normal([fc_1, fc_2]))
	b_fc_2 = tf.Variable(tf.zeros([fc_2]))

	w_output = tf.Variable(tf.truncated_normal([fc_2, n_outputs]))
	b_output = tf.Variable(tf.zeros([n_outputs]))

	def forward(inputs, dropout=True):
		conv = tf.nn.conv2d(inputs, w_conv_1, strides=[1,1,1,1], padding='SAME')
		conv = tf.nn.relu(conv + b_conv_1)
		conv = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		conv = tf.nn.conv2d(conv, w_conv_2, strides=[1,1,1,1], padding='SAME')
		conv = tf.nn.relu(conv + b_conv_2)
		conv = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		conv = tf.reshape(conv, [-1, 7 * 7 * conv2])
		hidden_1 = tf.matmul(conv, w_fc_1) + b_fc_1
		hidden_1 = tf.nn.relu(hidden_1)
		if dropout:
			hidden_1 = tf.nn.dropout(hidden_1, 0.5)
		hidden_2 = tf.matmul(hidden_1, w_fc_2) + b_fc_2
		hidden_2 = tf.nn.relu(hidden_2)
		if dropout:
			hidden_2 = tf.nn.dropout(hidden_2, 0.5)
		return tf.matmul(hidden_2, w_output) + b_output

	output = forward(tf_train_data)
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(output, tf_train_labels))

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	train_prediction = tf.nn.softmax(output)
	valid_prediction = tf.nn.softmax(forward(tf_valid_data, False))
	test_prediction = tf.nn.softmax(forward(tf_test_data, False))

# Run NN.
num_steps = 101
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print "Initialized"
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_data[offset: offset+batch_size, :]
		batch_labels = train_labels[offset: offset+batch_size, :]
		feed_dict = {
			tf_train_data: batch_data,
			tf_train_labels: batch_labels
		}
		_, l, prediction = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)

		if(step % 10 == 0):
			print "Minibatch loss at step %d: %.6f" % (step, l)
			print "Minibatch Accuracy: %.2f" % calculate_accuracy(
				prediction, batch_labels)
			print "Valid Accuracy: %.2f" % calculate_accuracy(
				valid_prediction.eval(), valid_labels)
	print "Test accuracy: %.2f" % calculate_accuracy(
		test_prediction.eval(), test_labels)

print "Done"
