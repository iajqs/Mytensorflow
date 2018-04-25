from datetime import datetime
import math
import time
import tensorflow as tf

#the size of the every batch
batch_size = 32
#the number of batch
num_batches = 100

#print the layer name and size
def print_activations(t):
    print(t.op.name, '' , t.get_shape().as_list)


def inference(images):
    parameters = []
    # the First Convolute level
    with tf.name_scope('conv1') as scope:
        ##the Convolution kernel's size is 11x11
        ##the feature's number is 3
        ##the number of Convolution kernel is 64
        ##the convolute operation
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                dtype=tf.float32, stddev=1e-1), name='weights')
        ##run the convolute operation
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        ##get the biases
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        ##print the infomation of conv1
        print_activations(conv1)
        ##add the kernel, biases which can train to the parameters
        parameters += [kernel, biases]

#the LRN layer
lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
#the max pool
pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='VALID', name='pool1')
#print the infomation of pool1
print_activations(pool1)

#######################################################################
#                                                                     #
#                                                                     #
# i think the next content is not important or repeated, so i needn't #
# write the code again                                                #
#                                                                     #
#######################################################################