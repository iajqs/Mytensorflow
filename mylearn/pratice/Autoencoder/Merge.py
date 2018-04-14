import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#make the weight no height or light
##fan_in is the number of input node
##fan_out is the number of output node
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    ###come on, i can't understand this function
    ###--tf.random_uniform((x,y),minval=low,maxval=high,dtype=tf.float32)))
    ###-- return a (x*y)matrix, the value is between low and high, the values are uniformly distributed
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    ##n_input is the number of the variable
    ##n_hidden is the Hidden layer node number
    ##transfer_function is the Hidden layer's Activation function
    ##optimizer
    ##scale is the coefficient of the noise
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initializer_weights()
        self.weigths = network_weights

        # x is the trainData
        ##placeholder as the place for data input
        self.x = tf.placeholder(tf.float32,[None, self.n_input])
        ##tf.matnul is Matrix multiplication
        ##tf.add just is add
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x+scale * tf.random_normal((n_input,)),
            self.weigths['w1']), self.weigths['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weigths['w2']), self.weigths['b2'])

        #define a lose fuction
        ##the Squared Error as the cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))   #tf.reduce_mean
        ##use the optimizer to optimize the cost
        self.optimizer = optimizer.minimize(self.cost)

        ##create a Session and init all variables of the model
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    #initalize the werghts
    def _initializer_weights(self):
        ##dict can create a dictionary and return a dictionary
        ###dict(a='a', b='b', t='t')  # 传入关键字
        ###{'a': 'a', 'b': 'b', 't': 't'}
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        ##tf.zeros(shape[], dtype=tf.float32, name=None)
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                     self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype = tf.float32))

        return all_weights

    # caculate the cost and run the optimizer
    def partial_fit(self, X):
        ##get the self.cost and self.optimizer as result of the feed_dict is the {self.x: X, self.scale: self.training_scale}
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    #caculate the cost
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
                                                     self.scale: self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weigths["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weigths['w1'])

    def getBiases(self):
        return self.sess.run(self.weigths['b1'])






#here is test code
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

##preprocessing the data about the train and the test
def standard_scale(X_train, X_test):
    preprocesser = prep.StandardScaler().fit(X_train)
    X_train = preprocesser.transform(X_train)
    X_test = preprocesser.transform(X_test)
    return X_train, X_test

##get the dataBlock from data by random, and never put back
##all in all, I don't know why this fuction have the fuction like never put back
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index+batch_size)]

##get the data about the train and the test
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

##get the samples numbers of train_data
n_samples = int(mnist.train.num_examples)
##the Max training times
training_epochs = 20
##any random block size
batch_size = 128
##every epoch display the cost once
display_step = 1
##create a AGN, and put the variables....
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale = 0.01)

for epoch in range(training_epochs):
    avg_cost = 0;
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(mnist.train.images, batch_size) # X_train

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:",'%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(mnist.test.images))) #X_test





sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                             reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entroy)

tf.global_variables_initializer().run()

X_test = autoencoder.reconstruct(X_test)
for i in range(3000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    batch_xs = autoencoder.reconstruct(batch_xs)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: X_test, y_:mnist.test.labels,
                     keep_prob: 1.0}))


