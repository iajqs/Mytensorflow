import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# download and load the data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# set the config of train
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

#set the layer size
## the images' width
n_input = 28
## the unrolled steps of LSTM, it also is the height of images
n_steps = 28
## the number of hidden node of LSTM
n_hidden = 256
## the result(lables) number
n_classes =10

# create the placeholder for x and y
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# create the weights and biases
weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

# create the BiRNN
def BiRNN(x, weights, biases):

    # # processing the x
    # # # translate the [0, 1, 2] to [1, 0, 2],
    # # # struction from [batch_size, n_steps, n_input] to [n_steps, batch_size, n_input]
    x = tf.transpose(x, [1, 0, 2])
    # # translate the struction from [n_steps, batch_size, n_input] to [n_steps x batch_size, n_input]
    x = tf.reshape(x, [-1, n_input])
    # # use the tf.split to split the x to be a list which length is n_steps
    # # # and size of every tensor which belong to list is (batch_size, n_input)
    x = tf.split(x, n_steps)

    # # the forward lstm cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    # # the backward lstm cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    # # the model composite by forward lstm cell and backward lstm cell
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                        lstm_bw_cell, x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

# get the predicted result
pred = BiRNN(x, weights, biases)
# caculate the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=y))
# create the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# caculate the correct rate
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# get the mean
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# init the variables
init = tf.global_variables_initializer()

# training time
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # # # this step is a little weird, because at the function BiRNN, it translate that back
        # # # ok, I think I know the reason of it, this step translate the origin data first
        # # # and then, at the function,
        # # # it translate the translated data is not the same with translate the origin data direct
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y:batch_y})

            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))