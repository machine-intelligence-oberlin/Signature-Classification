'''
TutorialsPoint Recurrent Neural Network in TensorFlow

This tutorial demonstrates how to create an RNN in TensorFlow
for use on the MNIST dataset.

Comments have been added for how we will use this RNN to classify
the identity of people based on their signatures. Making this
model work for us is the first step in understanding TensorFlow and
machine learning.
'''

from __future__ import print_function 
# see https://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data 


# REPLACE THIS LINE
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) 
# Loads in signature data, reduces image size, and converts to one hot
# Code Here


# REPLACE THESE LINES
n_input = 28 
n_steps = 28
n_hidden = 128
n_classes = 10
# n_input by n_steps pixels, n_classes = num types of signatures (num people)


# ADD COMMENTS EXPLAINING WHAT IS HAPPENING
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes]
weights = {
   'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
   'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
   x = tf.unstack(x, n_steps, 1)
   lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
   outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
   return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()


with tf.Session() as sess:
   sess.run(init)
   step = 1
   while step * batch_size < training_iters:
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      batch_x = batch_x.reshape((batch_size, n_steps, n_input))
      sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
      if step % display_step == 0:
      acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
      loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

      # Original
      #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
      #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
      #      "{:.5f}".format(acc))
      # Edited
      print(f"Iter: {str(step*batch_size)} Minibatch Loss: {loss} Training Accuracy: {acc}")
   step += 1
   print("Optimization Finished!")
      test_len = 128
   test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
   test_label = mnist.test.labels[:test_len]
   print("Testing Accuracy:", \
      sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
