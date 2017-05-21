#!/usr/bin/env python
# coding: utf-8

# referenced http://qiita.com/mokemokechicken/items/8216aaad36709b6f0b5c


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H =100
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5


# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
#W = weight_variable((784, H))
#b1 = bias_variable([H])
W = tf.Variable(tf.random_normal([784, H], stddev=0.1), name="W")
b1 = tf.Variable(tf.zeros([H]), name="b1")


# Hidden Layer: h
# softsign(x) = x / (abs(x)+1); https://www.google.co.jp/search?q=x+%2F+(abs(x)%2B1)
h = tf.nn.softsign(tf.matmul(x, W) + b1)
h = tf.nn.relu(h)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W, name="W2")  # 転置
b2 = tf.Variable(tf.zeros([784]), name="b2")
y = tf.nn.relu(tf.matmul(h, W2) + b2)

# Define Loss Function
p1 = tf.reduce_mean(h, 0, name='p1')
p = 0.005
KL = p*tf.log(p/p1) + (1-p)*tf.log((1-p)/(1-p1))
KL_reg = 0.3*tf.reduce_sum(KL, name="KL")
loss =  KL_reg + tf.nn.l2_loss(y - x) / BATCH_SIZE

# For tensorboard learning monitoring
tf.summary.scalar("l2_loss", loss)
tf.summary.scalar("KL_reg", KL_reg)
tf.summary.image("W", tf.reshape(W2, [-1, 28, 28, 1]), H)
tf.summary.image("input", tf.reshape(x, [-1, 28, 28, 1]), 16)
tf.summary.image("output", tf.reshape(y, [-1, 28, 28, 1]), 16)
# Use Adam Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter('summarys/loss', graph_def=sess.graph_def)

# Training
for step in range(4000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, keep_prob: (1-DROP_OUT_RATE)})
    # Collect Summary
    summary_op = tf.summary.merge_all()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs, keep_prob: 1.})
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        _data = sess.run([loss, KL_reg], feed_dict={x: batch_xs, keep_prob: 1.})
        print("loss {0} \t  KL regulazation {1}".format(_data[0], _data[1]))

