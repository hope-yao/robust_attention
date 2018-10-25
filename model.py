"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
from glimpse import take_a_2d_glimpse
from multiGPU_utils import *

class Model_att():
    def __init__(self, x, y, cfg):
        self.x_input = x
        self.y_input = y
        self.num_glimpse = cfg['num_glimpse']
        self.glimpse_size = cfg['glimpse_size']
        learning_rate = cfg['lr']
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

        self.presoftmax_voting = []
        self.x_crop = []
        self.adv_grad = []
        self. xent = []
        tower_grads=[]
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in xrange(2):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('TOWER', i)) as scope:
                        x_input_i = self.x_input[i*4:(i+1)*4]
                        y_input_i = self.y_input[i*4:(i+1)*4]
                        if 0:
                            with tf.variable_scope('att_ctr_pred') as scope:
                                with slim.arg_scope([slim.conv2d], kernel_size=3):
                                    x = slim.conv2d(x_input_i, num_outputs=16, scope='conv1')
                                    x = slim.max_pool2d(x, kernel_size=2, scope='pool1')
                                    x = slim.conv2d(x, num_outputs=32, scope='conv2')
                                    x = slim.max_pool2d(x, kernel_size=2, scope='pool2')
                                    x = slim.flatten(x, scope='flatten')
                                    x = slim.fully_connected(x, num_outputs=2 * self.num_glimpse, activation_fn=None, scope='fc1')

                        else:
                            x = tf.random_uniform((x_input_i.get_shape().as_list()[0], 2 * self.num_glimpse), 0, 1)
                        loc = tf.split(x, self.num_glimpse, axis=1)

                        presoftmax_voting_i = []
                        crop_i = []
                        with tf.variable_scope('classifier') as scope:
                            for i, loc_i in enumerate(loc):
                                # crop
                                x_crop  = []
                                for j in range(3):
                                    x_in = x_input_i[:, :, :, j]
                                    x_out = take_a_2d_glimpse(x_in, loc_i, self.glimpse_size, delta=1, sigma=0.4)
                                    x_crop  += [tf.expand_dims(x_out, -1)]
                                x_crop = tf.concat(x_crop, -1)
                                crop_i += [tf.expand_dims(x_crop, 0)]
                                x = slim.conv2d(x_crop, kernel_size=5, num_outputs=32, scope='conv1')
                                x = slim.max_pool2d(x, kernel_size=2)
                                x = slim.conv2d(x, kernel_size=5, num_outputs=64, scope='conv2')
                                x = slim.max_pool2d(x, kernel_size=2)
                                x = slim.flatten(x, scope='flatten')
                                x = slim.fully_connected(x, num_outputs=1024, scope='fc1')
                                x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')
                                tf.get_variable_scope().reuse_variables()
                                presoftmax_voting_i += [x]
                                tf.get_variable_scope().reuse_variables()
                                assert tf.get_variable_scope().reuse == True

                tf.get_variable_scope().reuse_variables()
                presoftmax_voting = tf.reduce_mean(presoftmax_voting_i, 0)
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input_i, logits=presoftmax_voting)
                xent = tf.reduce_sum(y_xent)
                # adversarial grads across GPU
                adv_grad_i = tf.gradients(xent, x_input_i)[0]
                self.adv_grad += [adv_grad_i]
                # training grads across GPU.
                grad_i = opt.compute_gradients(xent)
                tower_grads.append(grad_i)

                self.presoftmax_voting += [presoftmax_voting]
                self.x_crop += [tf.concat(crop_i,0)]
                self.xent += [xent]

        self.adv_grad = tf.concat(self.adv_grad, 0)
        self.x_crop = tf.transpose(tf.concat(self.x_crop, 1), (1, 0, 2, 3, 4)) # batch, att, x, y, ch
        self.pre_softmax = tf.concat(self.presoftmax_voting, 0) # this is the most easy might not be the best way to do voting
        self.y_pred = tf.argmax(self.pre_softmax, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.y_input), tf.float32))
        self.xent = tf.reduce_mean(self.xent)
        grads = average_gradients(tower_grads)
        self.train_op = opt.apply_gradients(grads)
        print('initialization done....')



class Model_madry(object):
  def __init__(self, x):
    self.x_image = x

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')