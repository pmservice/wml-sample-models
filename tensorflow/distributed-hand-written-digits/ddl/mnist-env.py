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

import os, sys
import numpy as np
import tempfile
import ddl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100, "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data", "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

#
FLAGS = tf.app.flags.FLAGS
data_dir = os.getenv["DATA_DIR"]
result_dir = os.getenv("RESULT_DIR")
model_path = result_dir + "/model"

#
def main(_):
   # create a spec from the parameter server and worker hosts
   ps_hosts = FLAGS.ps_hosts.split(",")
   worker_hosts = FLAGS.worker_hosts.split(",")
   cluster = tf.train.ClusterSpec({"ps": ps_hosts,"worker": worker_hosts})
   # create a server for the local task
   server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

   # parameter server
   if FLAGS.job_name == "ps":
     print("parameter server on " + server.target.decode("utf-8"))
     sys.stdout.flush()
     server.join()  # this never finishes
     return

   # otherwise this is a worker
   if FLAGS.job_name != "worker":
     print("invalid job_name="+FLAGS.job_name)
     return
   print("worker %d started on %s" % (FLAGS.task_index,server.target.decode("utf-8")))

   # load mnist data files
   mnist = input_data.read_data_sets(data_dir, one_hot=True)
   # only one checkpoint dir needed per task
   checkpoint_dir = "%s/train_logs%d" % (result_dir,FLAGS.task_index)
   print(checkpoint_dir)
   sys.stdout.flush()

   # build the model
   with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
      W = tf.Variable(tf.zeros([784,10]), name="weights")
      b = tf.Variable(tf.zeros([10]), name="bias")
      x = tf.placeholder(tf.float32, [None,784], name="x_input")
      y = tf.placeholder(tf.float32, [None,10], name="y_output")
      model = tf.add(tf.matmul(x, W), b, name="model") # y = Wx + b
      # these are used for training
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model))
      prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
      accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")
      global_step = tf.train.get_or_create_global_step()
      train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost, global_step=global_step)

   # start the training session
   hooks = [tf.train.StopAtStepHook(last_step=501)]
   with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index==0), checkpoint_dir=checkpoint_dir, hooks=hooks) as sess:
      # loop through training data
      while not sess.should_stop():
         batch_xs, batch_ys = mnist.train.next_batch(1000)  # session divides up the training data
         _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y: batch_ys})
         if (step % 10 == 0) and (not sess.should_stop()):
            loss, acc = sess.run([cost,accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
            print("{:4d}".format(step) + ": " + "{:.6f}".format(loss) + ", accuracy=" + "{:.5f}".format(acc))
            sys.stdout.flush()

   # we are done (if we are not chief)
   if FLAGS.task_index != 0:
      return
   # the remainder code tests the model and saves it

   # get the last checkpoint file which has the completed parameters
   checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
   meta_file = checkpoint_file + ".meta"
   print(meta_file)
   sys.stdout.flush()

   # output the model for inference/serving
   with tf.Graph().as_default(): # use a new graph -- or tf.reset_default_graph() may work as well
      config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
      with tf.Session(config=config) as sess:
         # restoring the model from the last checkpoint file
         saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
         saver.restore(sess, checkpoint_file)
         # compute and print the accuracy based on the test data
         acc = sess.run("accuracy:0", feed_dict={"x_input:0": mnist.test.images, "y_output:0": mnist.test.labels})
         print("Test accuracy = "+"{:5f}".format(acc))
         # retrieve some of the graph elements we need to reference in the model builder signature
         x = tf.get_default_graph().get_tensor_by_name("x_input:0")
         model = tf.get_default_graph().get_tensor_by_name("model:0")
         predictor = tf.argmax(model, 1, name="predictor")
         # create the model builder
         inputs_classes = tf.saved_model.utils.build_tensor_info(x)           # input an image
         outputs_classes = tf.saved_model.utils.build_tensor_info(predictor)  # output its class (0-9)
         signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS:inputs_classes},outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:outputs_classes},method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
         builder = tf.saved_model.builder.SavedModelBuilder(model_path)       # path to store model
         legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
         # add the graph and save the model
         builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],signature_def_map={'predict_images': signature},legacy_init_op=legacy_init_op)
         save_path = builder.save()
         print("Model saved in file: %s" % save_path.decode("utf-8"))

#
if __name__ == "__main__":
   tf.app.run()

