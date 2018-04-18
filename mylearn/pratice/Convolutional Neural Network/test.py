import numpy as n
import tensorflow as tf

#test for ComplexOne's first problem
t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(t)

reshape = tf.reshape(t, [3,-1])
dim = reshape.get_shape()[1].value
print(reshape)