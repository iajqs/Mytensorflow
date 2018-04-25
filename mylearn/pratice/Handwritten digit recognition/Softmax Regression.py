import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#load the MINST
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#register a session as the eviroment for caculate
sess = tf.InteractiveSession()
#x is the trainData
##placeholder as the place for data input
x = tf.placeholder(tf.float32, [None, 784])
#set the weight, the feature's number is 784, the label's number is 10
w = tf.Variable(tf.zeros([784, 10]))
#set the bia, the label's number is 10, so set 10 bias too.
b = tf.Variable(tf.zeros([10]))
#the train model
##y is the trainResult
##y = softmax(Wx+b) in TensorFlow the y as the result of this formula
y = tf.nn.softmax(tf.matmul(x, w) + b)
#cross-entropy in TensorFlow
##y_ is the real label
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#once train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#initialize the global variables
tf.global_variables_initializer().run()
#train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    ##this x is for the upper x
    ##this y_ is for the upper y_
    train_step.run({x:batch_xs,y_:batch_ys})


#this y is come from upper y, and the upper y is come from the train_step.run, is the python is caculate from the last to first line?
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#caculate the accuracy
## the tf.cast is translate the bool to float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print the test result
##this x is the x first time input the system
##the y_ too
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
