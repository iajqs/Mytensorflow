from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#get the data from MNIST_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#register a session as the eviroment for caculate
sess = tf.InteractiveSession()

#the input_data's  dimension number
in_units = 784
#the dimension number of hidden level
h1_units = 300

#the weights[0]
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
#the bias[0]
b1 = tf.Variable(tf.zeros([h1_units]))
#the weights[1]
W2 = tf.Variable(tf.zeros([h1_units, 10]))
#the bias[1]
b2 = tf.Variable(tf.zeros([10]))

#the data which need input from run{x: }
x = tf.placeholder(tf.float32, [None, in_units])
#the data which need input from run{keep_prob}
keep_prob = tf.placeholder(tf.float32)

#the train model
##into the hidden1
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
##Droput: random make some node to 0
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
##get the output from hidden1 as the result
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
#the true result from input
y_ = tf.placeholder(tf.float32, [None, 10])
#caculate the cost
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                             reduction_indices=[1]))
#once train
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entroy)
#initialize the global variables
tf.global_variables_initializer().run()
#the train time
for i in range(3000):
    ## get the batch from mnist by random
    batch_xs , batch_ys = mnist.train.next_batch(100)
    ##once train and input the data like x, y, keep_prob
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

#caculate the result from y compare to y_
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
#caculate the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#run and get accuracy by this function and input the x, y, keep_prob
print(accuracy.eval({x: mnist.test.images, y_:mnist.test.labels,
                     keep_prob: 1.0}))




