"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
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
        self.nGPU = cfg['nGPU']
        sub_batch_size = self.sub_batch_size = cfg['batch_size'] // cfg['nGPU']
        learning_rate = cfg['lr']
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

        self.presoftmax_voting = []
        self.x_crop = []
        self.adv_grad = []
        self. xent = []
        tower_grads=[]
        self.x_crop = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in xrange(self.nGPU):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('TOWER', i)) as scope:
                        x_input_i = self.x_input[i*sub_batch_size:(i+1)*sub_batch_size]
                        y_input_i = self.y_input[i*sub_batch_size:(i+1)*sub_batch_size]
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
                        with tf.variable_scope('classifier') as scope:
                            x_crop_minibatch = []
                            for i, loc_i in enumerate(loc):
                                # crop
                                x_crop_minibatch_loci  = []
                                for j in range(3):
                                    x_in = x_input_i[:, :, :, j]
                                    x_out = take_a_2d_glimpse(x_in, loc_i, self.glimpse_size, delta=1, sigma=0.4)
                                    x_crop_minibatch_loci += [tf.expand_dims(x_out, -1)]
                                x_crop_minibatch_loci = tf.concat(x_crop_minibatch_loci, -1)
                                x_crop_minibatch += [tf.expand_dims(x_crop_minibatch_loci, 0)]

                                h1 = x = slim.conv2d(x_crop_minibatch_loci, kernel_size=5, num_outputs=16, scope='conv1')
                                x = slim.conv2d(x, kernel_size=5, num_outputs=16, scope='conv11')
                                x += h1
                                h2 = x = slim.conv2d(x, kernel_size=5, num_outputs=160, scope='conv2')
                                x = slim.conv2d(x, kernel_size=5, num_outputs=160, scope='conv21')
                                x += h2
                                x = slim.max_pool2d(x, kernel_size=2)
                                h3 = x = slim.conv2d(x, kernel_size=5, num_outputs=320, scope='conv3')
                                x = slim.conv2d(x, kernel_size=5, num_outputs=320, scope='conv31')
                                x += h3
                                x = slim.max_pool2d(x, kernel_size=2)
                                h4 = x = slim.conv2d(x, kernel_size=5, num_outputs=640, scope='conv4')
                                x = slim.conv2d(x, kernel_size=5, num_outputs=640, scope='conv41')
                                x += h4
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
                adv_grad_minibatch = tf.gradients(xent, x_input_i)[0]
                self.adv_grad += [adv_grad_minibatch]
                # training grads across GPU.
                grad_i = opt.compute_gradients(xent)
                tower_grads.append(grad_i)

                self.presoftmax_voting += [presoftmax_voting]
                self.x_crop += [tf.concat(x_crop_minibatch,0)]
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


class Model_Madry(object):
  """ResNet model."""

  def __init__(self, x, y, cfg, mode):
    """ResNet constructor.
    Args:
      mode: One of 'train' and 'eval'.
    """

    self.x_input = x
    self.y_input = y
    learning_rate = cfg['lr']
    self.opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    self.mode = mode
    self._build_model()

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, 10)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

    self.adv_grad = tf.gradients(self.xent, self.x_input)[0]
    grads = self.opt.compute_gradients(self.xent)
    self.train_op = self.opt.apply_gradients(grads)

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


