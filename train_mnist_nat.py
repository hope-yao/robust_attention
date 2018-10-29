import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model_mnist import Model_att as Model_att
from model_mnist import Model_madry as Model_madry
from viz2 import *
import os
import numpy as np
from utils import creat_dir
from tqdm import tqdm
slim = tf.contrib.slim

def main(cfg):
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    lr = cfg['lr']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    # build classifier
    model = Model_att(input_images, input_label, glimpse_size, num_glimpse)


    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(model.xent)
    train_op=optimizer.apply_gradients(grads)
    saver = tf.train.Saver()
    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    hist = {'train_acc': [],
            'train_adv_acc': [],
            'test_acc': [],
            'test_adv_acc': [],
            'train_loss': [],
            'test_loss': [],
            'train_adv_loss': [],
            'test_adv_loss': []}
    train_iters=500000
    for itr in tqdm(range(train_iters)):
        x_batch_train, y_batch_train = mnist.train.next_batch(batch_size)
        nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 1),
                          input_label: y_batch_train}
        sess.run(train_op, feed_dict=nat_dict_train)

        if itr % 100 == 0:
            train_acc_i, train_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_train)
            x_batch_test, y_batch_test = mnist.test.next_batch(batch_size)
            nat_dict_test = {input_images: x_batch_test.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_test}
            test_acc_i, test_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_test)
            print("iter: {}, train_acc:{}  test_acc:{} train_loss:{}  test_loss:{} "
                  .format(itr, train_acc_i, test_acc_i, train_loss_i, test_loss_i))

            hist['train_acc'] += [train_acc_i]
            hist['test_acc'] += [test_acc_i]
            hist['train_loss'] += [train_loss_i]
            hist['test_loss'] += [test_loss_i]
            np.save('hist',hist)
            saver.save(sess,'nat_ckpt')
    print('done')


if __name__ == "__main__":


    cfg = {'batch_size': 32,
           'img_dim': 2,
           'img_size': 28,
           'num_glimpse': 5,
           'glimpse_size': 20,
           'lr': 1e-4
           }
    main(cfg)
