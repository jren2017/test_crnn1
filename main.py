import tensorflow as tf


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


ii = tf.constant([[1, 0, 2, 3, 0, 1, 1],[1, 0, 2, 3, 0, 1, 1]], dtype=tf.float32, name='i')

label = 1 # for classification

k2 = tf.constant([[2, 1, 3], [2, 1, 3]], dtype=tf.float32, name='k')

data = tf.reshape(ii, [1, int(ii.shape[1]), 2], name='data')

kernel = tf.reshape(k2, [1, int(k2.shape[0]), 3], name='kernel')

# res2 = crnn(data, 4, 1, 10, 1, 'simple', True, 0.1, 'SAME', 'test_crnn')
res2 = crnn(data, 4, 1, 10, 1, 'simple', False, 0.1, 'VALID', 'test_crnn') # output dimension, 4x10

res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'VALID'))

num_steps = 4
batch_size = 1
num_classes = 1
state_size = 10
learning_rate = 0.1

init_state = tf.zeros([batch_size, state_size])

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, [res2], initial_state=init_state) 
# How to input the output of the crnn into the next rnn model, to do classification, the label of our only one example is "1" 
# or we can generate many similar examples like this


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(res))
    print(sess.run(res2))
    print(rnn_outputs)


