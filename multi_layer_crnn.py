import tensorflow as tf

import numpy as np


# res2 = crnn(data, 4, 1, 10, 1, 'simple', True, 0.1, 'SAME', 'test_crnn')
# crnn(X_in)
# X_in = crnn(X_in, 4, 1, 10, 1, 'simple', False, 0.1, 'VALID', 'test_crnn') # output dimension, 4x10
# CRNN layer
def crnn(tensor, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std, padding, scope_name):
	with tf.variable_scope(scope_name, initializer=tf.truncated_normal_initializer(stddev=w_std)):
		# Expand to have 4 dimensions if needed
		if len(tensor.shape) == 3:
			tensor = tf.expand_dims(tensor, 3)

		# Extract the patches (returns [batch, time-steps, 1, patch content flattened])
		batch_size = tensor.shape[0].value
		n_in_features = tensor.shape[2].value
		patches = tf.extract_image_patches(images=tensor,
		                                   ksizes=[1, kernel_size, n_in_features, 1],
		                                   strides=[1, stride, n_in_features, 1],
		                                   rates=[1, 1, 1, 1],
		                                   padding=padding)
		patches = patches[:, :, 0, :]

		# Reshape to do:
		# 1) reshape the flattened patches back to [kernel_size, n_in_features]
		# 2) combine the batch and time-steps dimensions (which will be the new 'batch' size, for the RNN)
		# now shape will be [batch * time-steps, kernel_size, n_features]
		time_steps_after_stride = patches.shape[1].value
		patches = tf.reshape(patches, [batch_size * time_steps_after_stride, kernel_size, n_in_features])

		# Transpose and convert to a list, to fit the tf.contrib.rnn.static_rnn requirements
		# Now will be a list of length kernel_size, each element of shape [batch * time-steps, n_features]
		patches = tf.unstack(tf.transpose(patches, [1, 0, 2]))

		# Create the RNN Cell
		if rnn_type == 'simple':
			rnn_cell_func = tf.contrib.rnn.BasicRNNCell
		elif rnn_type == 'lstm':
			rnn_cell_func = tf.contrib.rnn.LSTMBlockCell
		elif rnn_type == 'gru':
			rnn_cell_func = tf.contrib.rnn.GRUBlockCell
		if not bidirectional:
			rnn_cell = rnn_cell_func(out_channels)
		else:
			rnn_cell_f = rnn_cell_func(out_channels)
			rnn_cell_b = rnn_cell_func(out_channels)

		# Multilayer RNN? (does not appear in the original paper)
		if rnn_n_layers > 1:
			if not bidirectional:
				rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_n_layers)
			else:
				rnn_cell_f = tf.contrib.rnn.MultiRNNCell([rnn_cell_f] * rnn_n_layers)
				rnn_cell_b = tf.contrib.rnn.MultiRNNCell([rnn_cell_b] * rnn_n_layers)

		# The RNN itself
		if not bidirectional:
			outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, patches, dtype=tf.float32)
		else:
			outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(rnn_cell_f, rnn_cell_b,
			                                                                                    patches,
			                                                                                    dtype=tf.float32)

		# Use only the output of the last time-step (shape will be [batch * time-steps, out_channels]).
		# In the case of a bidirectional RNN, we want to take the last time-step of the forward RNN,
		# and the first time-step of the backward RNN.
		if not bidirectional:
			outputs = outputs[-1]
		else:
			half = int(outputs[0].shape.as_list()[-1] / 2)
			outputs = tf.concat([outputs[-1][:, :half],
			                     outputs[0][:, half:]],
			                    axis=1)

		# Expand the batch * time-steps back (shape will be [batch_size, time_steps, out_channels]
		if bidirectional:
			out_channels = 2 * out_channels
		outputs = tf.reshape(outputs, [batch_size, time_steps_after_stride, out_channels])

		return outputs


# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
	X_signals = []

	for signal_type_path in X_signals_paths:
		file = open(signal_type_path, 'r')
		# Read dataset from disk, dealing with text files' syntax
		X_signals.append(
			[np.array(serie, dtype=np.float32) for serie in [
				row.replace('  ', ' ').strip().split(' ') for row in file
			]]
		)
		file.close()

	return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
	file = open(y_path, 'r')
	# Read dataset from disk, dealing with text file's syntax
	y_ = np.array(
		[elem for elem in [
			row.replace('  ', ' ').strip().split(' ') for row in file
		]],
		dtype=np.int32
	)
	file.close()
	# Substract 1 to each output class for friendly 0-based indexing
	return y_ - 1


class Config(object):
	"""
	define a class to store parameters,
	the input should be feature mat of training and testing

	Note: it would be more interesting to use a HyperOpt search space:
	https://github.com/hyperopt/hyperopt
	"""

	def __init__(self, X_train, X_test):
		# Input data
		self.train_count = len(X_train)  # 7352 training series
		self.test_data_count = len(X_test)  # 2947 testing series
		self.n_steps = len(X_train[0])  # 128 time_steps per series

		# Training
		self.learning_rate = 0.0025
		self.lambda_loss_amount = 0.0015
		self.training_epochs = 500
		self.batch_size = 1500

		# LSTM structure
		self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
		self.n_hidden = 32  # nb of neurons inside the neural network
		self.n_hidden1 = 20  # nb of neurons inside the neural network
		self.n_classes = 6  # Final output classes
		self.W = {
			'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
			'output': tf.Variable(tf.random_normal([self.n_hidden1, self.n_classes]))
		}
		self.biases = {
			'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
			'output': tf.Variable(tf.random_normal([self.n_classes]))
		}


def LSTM_CRNN_Network(XX, config, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std,
                      padding, scope_name):
	lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)


def LSTM_Network(_X, config):
	with tf.variable_scope('lstm1'):
		# (NOTE: This step could be greatly optimised by shaping the dataset once
		# input shape: (batch_size, n_steps, n_input)
		# _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
		# Reshape to prepare input to hidden activation
		# _X = tf.reshape(_X, [-1, config.n_inputs])
		# new shape: (n_steps*batch_size, n_input)

		# Linear activation
		# _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
		# Split data because rnn cell needs a list of inputs for the RNN inner loop
		# _X = tf.split(_X, config.n_steps, 0)

		# _X_converted = tf.(_X, dtype=tf.float32)
		# _X_converted = tf.convert_to_tensor(_X, dtype=tf.float64)
		# new shape: n_steps * (batch_size, n_hidden)
		# X_in = crnn(_X, 5, 1, 25, 1, 'simple', False, 0.1, 'SAME', 'test_crnn')  # output dimension, 4x10
		# X_in = crnn(_X, 5, 1, 25, 1, 'lstm', False, 0.1, 'SAME', 'test_crnn')  # output dimension, 4x10
		X_in = crnn(_X, 20, 1, 7, 1, 'gru', False, 0.1, 'SAME', 'test_crnn')  # output dimension, 4x10
		_X = tf.transpose(X_in, [1, 0, 2])
		# _X = tf.reshape(_X, [-1, config.n_inputs])
		_X = tf.reshape(_X, [-1, 7])
		_X = tf.split(_X, config.n_steps, 0)

		# Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
		lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
		lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)

		lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
		# tf.contrib.rnn.BasicLSTMCell()
		# Get LSTM cell output
		outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

		# X_out = outputs
		# X_out = tf.transpose(X_out, [1, 0, 2])
		# # _X = tf.reshape(_X, [-1, config.n_inputs])
		# X_out = tf.reshape(X_out, [-1, 20])
		# outputs = tf.split(X_out, config.n_steps, 0)
		outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)

	with tf.variable_scope('lstm2'):
		# (tensor, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std, padding, scope_name)
		X_out = crnn(outputs, 20, 1, 30, 1, 'gru', False, 0.1, 'SAME', 'test_crnn1')  # output dimension, 4x10
		X_out = tf.transpose(X_out, [1, 0, 2])
		# _X = tf.reshape(_X, [-1, config.n_inputs])
		X_out = tf.reshape(X_out, [-1, 30])
		X_out = tf.split(X_out, 128, 0)

		# ==========================
		# Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
		lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
		lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)

		lstm_cells1 = tf.contrib.rnn.MultiRNNCell([lstm_cell_3, lstm_cell_4], state_is_tuple=True)
		# tf.contrib.rnn.BasicLSTMCell()
		# Get LSTM cell output
		# array = W1.eval(sess)

		outputs1, states1 = tf.contrib.rnn.static_rnn(lstm_cells1, X_out, dtype=tf.float32)

		# X_out = outputs
		# X_out = tf.transpose(X_out, [1, 0, 2])
		# # _X = tf.reshape(_X, [-1, config.n_inputs])
		# X_out = tf.reshape(X_out, [-1, 20])
		# outputs = tf.split(X_out, config.n_steps, 0)
		outputs1 = tf.convert_to_tensor(outputs1, dtype=tf.float32)

		X_out1 = crnn(outputs1, 20, 1, 20, 1, 'gru', False, 0.1, 'SAME', 'test_crnn2')  # output dimension, 4x10

		# Get last time step's output feature for a "many to one" style classifier,
		# as in the image describing RNNs at the top of this page

		# lstm_last_output = outputs[-1]
		X_out1 = tf.convert_to_tensor(X_out1, dtype=tf.float32)
		lstm_last_output = X_out1[-14]
		print("=================")
		print(tf.shape(lstm_last_output))

		# Linear activation
	return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(y_):
	"""
	Function to encode output labels from number indexes.

	E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
	"""
	y_ = y_.reshape(len(y_))
	n_values = int(np.max(y_)) + 1
	return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

	INPUT_SIGNAL_TYPES = [
		"body_acc_x_",
		"body_acc_y_",
		"body_acc_z_",
		"body_gyro_x_",
		"body_gyro_y_",
		"body_gyro_z_",
		"total_acc_x_",
		"total_acc_y_",
		"total_acc_z_"
	]

	# Output classes to learn how to classify
	LABELS = [
		"WALKING",
		"WALKING_UPSTAIRS",
		"WALKING_DOWNSTAIRS",
		"SITTING",
		"STANDING",
		"LAYING"
	]

	DATA_PATH = "data/"
	DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
	print("\n" + "Dataset is now located at: " + DATASET_PATH)
	TRAIN = "train/"
	TEST = "test/"

	X_train_signals_paths = [
		DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
	]
	X_test_signals_paths = [
		DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
	]
	X_train = load_X(X_train_signals_paths)
	X_test = load_X(X_test_signals_paths)

	y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
	y_test_path = DATASET_PATH + TEST + "y_test.txt"
	y_train = one_hot(load_y(y_train_path))
	y_test = one_hot(load_y(y_test_path))

	# -----------------------------------
	# define parameters for model
	# -----------------------------------

	config = Config(X_train, X_test)
	print("Some useful info to get an insight on dataset's shape and normalisation:")
	print("features shape, labels shape, each features mean, each features standard deviation")
	print(X_test.shape, y_test.shape,
	      np.mean(X_test), np.std(X_test))
	print("the dataset is therefore properly normalised, as expected.")

	# ------------------------------------------------------
	# build the neural network
	# ------------------------------------------------------

	X = tf.placeholder(tf.float32, [1500, config.n_steps, config.n_inputs])
	Y = tf.placeholder(tf.float32, [1500, config.n_classes])
	print("___________________________")
	print(tf.shape(Y))

	pred_Y = LSTM_Network(X, config)

	# Loss,optimizer,evaluation
	l2 = config.lambda_loss_amount * \
	     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
	# Softmax loss and L2
	cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
	optimizer = tf.train.AdamOptimizer(
		learning_rate=config.learning_rate).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

	# --------------------------------------------
	# train the neural network
	# --------------------------------------------

	sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
	init = tf.global_variables_initializer()
	sess.run(init)

	best_accuracy = 0.0
	# Start training for each batch and loop epochs
	for i in range(config.training_epochs):
		for start, end in zip(range(0, config.train_count, config.batch_size),
		                      range(config.batch_size, config.train_count + 1, config.batch_size)):
			sess.run(optimizer, feed_dict={X: X_train[start:end],
			                               Y: y_train[start:end]})
		print(i)

		# Test completely at every epoch: calculate accuracy
		pred_out, accuracy_out, loss_out = sess.run(
			[pred_Y, accuracy, cost],
			feed_dict={
				X: X_test[0:1500],
				Y: y_test[0:1500]
			}
		)
		print("traing iter: {},".format(i) +
		      " test accuracy : {},".format(accuracy_out) +
		      " loss : {}".format(loss_out))
		best_accuracy = max(best_accuracy, accuracy_out)

	print("")
	print("final test accuracy: {}".format(accuracy_out))
	print("best epoch's test accuracy: {}".format(best_accuracy))
	print("")
