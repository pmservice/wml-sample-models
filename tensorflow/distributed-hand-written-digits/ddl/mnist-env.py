'''
Based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py:
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
Modifications:
*****************************************************************
Licensed Materials - Property of IBM
(C) Copyright IBM Corp. 2017. All Rights Reserved.
US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
*****************************************************************
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
import tempfile
import ddl
import os
from tensorflow.examples.tutorials.mnist import input_data

checkpoint_path = os.getenv("RESULT_DIR")
training_data_dir = os.getenv("DATA_DIR")
learner_id = os.getenv("LEARNER_ID")

#This can be removed after a PR on the ddl trainer is merged
#training_data_dir="/mnt/data/dlaas-ci-tf-training-data"

#print("checkpint path is {} and training_data_dir is {} and learner_id {}".format( checkpoint_path, training_data_dir, learner_id))
#if checkpoint_path == None:
#    exit(1)
#
if training_data_dir == None:
    exit(1)

#if learner_id == None:
#    exit(1)


FLAGS = None

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    ############################################################################
    # Import MNIST data
    ############################################################################
    mnist = input_data.read_data_sets(training_data_dir)

    # Parameters
    learning_rate = 0.001
    training_iters = 2000
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.int64, [None])

    # Construct model
    pred, keep_prob = deepnn(x)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred))

    with tf.name_scope('adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        objective = optimizer.minimize(cost)

    # Evaluate model
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), y)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # Launch the graph
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:

            ###################################################
            ### USE ddl.rank() and ddl.size() to load data  ###
            ###################################################
            batch_x, batch_y = mnist.train.next_batch(batch_size*ddl.size())

            #select one of partitions
            batch_x = np.split(batch_x,ddl.size())[ddl.rank()]
            batch_y = np.split(batch_y,ddl.size())[ddl.rank()]

            # Run optimization op (backprop)
            sess.run(objective, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("DDL " + str(ddl.rank()) + "] Iter " + str(step * batch_size) +
                  ", Minibatch Loss= " + "{:.6f}".format(loss) +
                  ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        print("DDL "+str(ddl.rank())+"] Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        print("DDL "+str(ddl.rank())+"] Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
