from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#construct some noise to break the Complete symmetry
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#add 0.1 to bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#cnn level
##x is the data input
##W is the  parameter for
    ###W[5,5,1,32]
        ####  Convolution kernel size is 5x5
        ####  the channel is 1, because the color just gray
        #### the number of Convolution is 32
##strides = [1, 1, 1, 1] Express handle all the digit
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

#let the 2x2 digit block become the 1x1 digit block
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding = 'SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
#let the 1x784 struction become the 28x28 struction
##the -1 represent the input size is not confirmed
##the new struction is 28x28
##the 1 represent the color channel just one
x_image = tf.reshape(x, [-1, 28, 28, 1])

#the first Convolution level
#init the parameter
##get the weights
    ###the Convolution kernel's size is 5x5
    ###the color channel is 1 (the feature's number just one)
    ###the number of Convolutino kernel is 32
W_conv1 = weight_variable([5, 5, 1, 32])
##get the bias
b_conv1 = bias_variable([32])
##run the convolute operation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
##let 2x2 digit block to 1x1
h_pool1 = max_pool_2x2(h_conv1)

#the second Convolution level
##get the weights
    ###the Convolution kernel's size is 5x5
    ###the feature's number just 32
    ###the number of Convolutino kernel is 64
W_conv2 = weight_variable([5, 5, 32, 64])
##get the bias
b_conv2 = bias_variable([64])
##run the convolute operation
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
##let 2x2 digit block to 1x1
h_pool2 = max_pool_2x2(h_conv2)
######## pass the Two convolution level, the data construct will be 7x7 and the size will be 7x7x64(the features' number)

#the full join level
##get the weights
W_fc1 = weight_variable([7*7*64, 1024])
##get the bias
b_fc1 = bias_variable([1024])
##translate the h_pool2(2D) result to 1D
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
##connect a full join level, tf.matmul is the matrix multiplication
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#the Dropout level
#throw some data to alleviate the Overfitting
##keep_prob is the Percentage that will be persist
keep_prob = tf.placeholder(tf.float32)
##throw!!! haha
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#let the output of Dropout level connect a Softmax level
##get the weights
W_fc2 = weight_variable([1024, 10])
##get the bias
b_fc2 = bias_variable([10])
##get the result by Softmax
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#the lose fuction
##the loss fuction
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
##train once
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#the fuction for evaluating the accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#the process about train
##init all variables
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1],
                                                    keep_prob: 1.0})
        print("strp %d, training accuracy %g" %(i, train_accuracy))

    train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob: 0.5})

#print the result of the test result
print("test accuracy %g" %accuracy.eval(feed_dict={
    x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
}))



