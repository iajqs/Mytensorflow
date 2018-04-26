from datetime import datetime
import math
import time
import tensorflow as tf

#the Convolutional Neural Network layer
def dev_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value


    with tf.name_scope(name) as scope:
        ##create the kernel para[kh, kw, n_in, n_out],
        ###kh: the height of the kernel
        ###kw: the width of the kernel
        ###n_in the channel's number of the input
        ###n_out the channel's number of the output
        kernel = tf.get_variable(scope + "w",
                shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer_conv2d())

        ##the Convolutional layer
        ###the step size is dh x dw
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
                            padding='SAME')
        ##init the bias
        bias_init_val = tf.constant.constant(0.0, shape=[n_out], dtype=tf.float32)
        ##chage the bias to be the changeable parameter
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        ##use the "tf.nn.bias_add" complete the "conv add biases"
        z = tf.nn.bias_add(conv, biases)
        ##use the relu to complete the nonlinerize operation
        activation = tf.nn.relu(z, name=scope)
        ##add the kernel biases which use for recreate the cnn to the parameter list p
        p += [kernel, biases]
        ##return the result
        return activation

#the complete connect layer
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        ## create the kernel para[n_in, n_out]
        ###n_in : the channel's number of the input
        ###n_out: the channel's number of the output
        kernel = tf.get_variable(scope + "w",
                shape=[n_in, n_out], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        ##create the changeadble biases = 0.1
        biases = tf.Variable(tf.constant(0.1, shape=[n_out],
                                         dtype=tf.float32), name='b')
        ##use the relu to complete the nonlinerize operation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        ##add the kernel biases which use for recreate the cnn to the parameter list p
        p += [kernel, biases]
        return activation

#the max pool
##size is kh x kw
##step is dh x dw
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


#this function is really same with the complex one
#I think I just complete the layer function create is enough
#because the next step is construct the layer building

#def inference_op(input_op, keep_prob):