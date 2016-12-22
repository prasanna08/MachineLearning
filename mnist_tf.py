"""Neural network built using tensorflow for MNIST data set."""
import tensorflow as tf
import numpy as np
import cPickle as pickle
import preprocess

def calculate_accuracy(prediction, target):
	return (100.0 * np.sum(prediction.argmax(axis=1) == target.argmax(axis=1))
			/ prediction.shape[0])


data = pickle.load(open('MNIST_dataset.pickle', 'rb'))
batch_size = 256
hidden_layer_size = 300

train_data = data['train_data']
train_labels = preprocess.onehot(data['train_labels'])
valid_data = data['valid_data']
valid_labels = preprocess.onehot(data['valid_labels'])
test_data = data['test_data']
test_labels = preprocess.onehot(data['test_labels'])

n_features = train_data.shape[1]
n_outputs = train_labels.shape[1]
graph = tf.Graph()

# Build NN.
with graph.as_default():
	tf_train_data = tf.placeholder(
		tf.float32, shape=(batch_size, n_features))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_outputs))
	tf_valid_data = tf.constant(data['valid_data'], dtype=tf.float32)
	tf_test_data = tf.constant(data['test_data'], dtype=tf.float32)

	w_hidden = tf.Variable(
		tf.truncated_normal([n_features, hidden_layer_size]))
	b_hidden = tf.Variable(tf.zeros([hidden_layer_size]))
	
	hidden_layer = tf.matmul(tf_train_data, w_hidden) + b_hidden
	hidden_layer = tf.sigmoid(hidden_layer)

	w_output = tf.Variable(
		tf.truncated_normal([hidden_layer_size, n_outputs]))
	b_output = tf.Variable(tf.zeros([n_outputs]))

	output = tf.matmul(hidden_layer, w_output) + b_output
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(output, tf_train_labels))

	optimizer = tf.train.AdagradOptimizer(0.3).minimize(loss)

	train_prediction = tf.nn.softmax(output)
	valid_prediction = tf.nn.softmax(
		tf.matmul(
			tf.sigmoid(tf.matmul(tf_valid_data, w_hidden) + b_hidden),
			w_output) + b_output)
	test_prediction = tf.nn.softmax(
		tf.matmul(
			tf.sigmoid(tf.matmul(tf_test_data, w_hidden) + b_hidden),
			w_output) + b_output)

# Run NN.
num_steps = 10001
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

		if(step % 500 == 0):
			print "Minibatch loss at step %d: %.6f" % (step, l)
			print "Minibatch Accuracy: %.2f" % calculate_accuracy(
				prediction, batch_labels)
			print "Valid Accuracy: %.2f" % calculate_accuracy(
				valid_prediction.eval(), valid_labels)
	print "Test accuracy: %.2f" % calculate_accuracy(
		test_prediction.eval(), test_labels)

print "Done"
