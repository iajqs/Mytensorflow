import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

#the times of the train
max_steps = 3000
#the size of the batch which is got from whole data by once
batch_size = 128
#the data path
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

#get the whole loss
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        ##tf.nn.l2_loss is the L2 loss
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#download and untie
cifar10.maybe_download_and_extract()

#Data Augmentation for train data
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir = data_dir, batch_size = batch_size
)

#get the test data
## cut 24x24 size block from the picture which is in the middle of picture
## data standlize
images_test, labels_test = cifar10_input.inputs(eval_data = True,
                                                data_dir = data_dir,
                                                batch_size = batch_size)

#the data input interface
image_holder = tf.placeholder(tf.float32,[batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#the First Convolute level
##get the weights
    ###the Convolution kernel's size is 5x5
    ###the feature's number is 3
    ###the number of Convolutino kernel is 64
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev = 5e-2,
                                    wl = 0.0)
##the convolute operation
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding = 'SAME')
##get the bias
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
##run the convolute operation
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
##let 3x3 digit block to 1x1 and move 2x2(ok, i don't know the move 2x2 mean what)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1],
                       padding='SAME')
##run the lrn level
norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha=0.001/9.0, beta = 0.75)

#the second Convolute level
##Attention: at the first cnn level, pool->norm; at the second cnn level, norm->pool
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2,
                                    wl = 0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha=0.001/9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

# the complete connect layer
# the first problem [batch_size, -1] and the easyOne's [-1 7*7*64] is the sanme thing?  done! see the test.py
#change the data struct to dimx384
##change the data struct to batch_size x (pool2.size/batch_size)
reshape = tf.reshape(pool2, [batch_size, -1])
##get the value about(pool2.size/batch_size)
dim = reshape.get_shape()[1].value
##get the weights (dim x 384)
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl = 0.004)
##get the bias (384x1)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
##use the relu function to make nonlinearization
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

#the complete connect layer
##get the weights (384 x 192)
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
##get the bias (192 x 1)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
##use the relu function to make nonlinearization
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)


#the complete connect layer
##get the weight (192 x 10)
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl = 0.0)
##get the bias (10 x 1)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
##get the result by matrix multiplication and matrix addition
logits = tf.add(tf.matmul(local4, weight5), bias5)


#get the loss by compare logits and lables
def loss(logits, labels):
    ##change the labels' dtype
    labels = tf.cast(labels, tf.int64)
    ##caculate the softmax and the cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels, name = 'cross_entropy_per_example'
    )
    ##get the mean of cross_entropy
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name = 'cross_entropy')
    ##add the "cross_entropy_mean" to the losses(the whole loss)
    tf.add_to_collection('losses', cross_entropy_mean)
    ##caculate the whole loss's sum, and return it
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

#get the loss, input the logits and the real labels
loss = loss(logits, label_holder)

#train once
##use the optimizer like AdamOptimizer, and the learning rate is 0.001
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#get the accuracy of top1
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

#create the session
sess = tf.InteractiveSession()
#init the global variables
tf.global_variables_initializer().run()

#start the thread line
tf.train.start_queue_runners()


#the training time
for step in range(max_steps):
    start_time = time.time()
    ##sess.run([a...],feed_dict={b....}),
    ### a... are the output data to the variabels which is before the equality sign,
    ### b... are the input data to the session
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder:image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1


precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)









