import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model_mnist import *
from viz2 import *
import os
import numpy as np
from utils import creat_dir
from tqdm import tqdm
from pgd_attack import LinfPGDAttack
slim = tf.contrib.slim

def main(cfg):
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    lr = cfg['lr']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 1))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    if 0:
        # build classifier
        model = Model_att(input_images, input_label, glimpse_size, num_glimpse)
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
        saver.restore(sess,'noatt_ckpt')
        mnist_adversary = np.load('mnist_adversary_noatt.npy').item()
        adv_x, adv_y = mnist_adversary['x'], mnist_adversary['y']
        overall_acc = []
        for itr in tqdm(range(300)):
            x_batch_test_adv, y_batch_test = adv_x[itr*batch_size:(itr+1)*batch_size], adv_y[itr*batch_size:(itr+1)*batch_size]
            adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_test}
            test_acc_i, test_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_test)
            overall_acc += [test_acc_i]
            print("iter: {}, test_acc:{}  test_loss:{} ".format(itr, test_acc_i, test_loss_i))
        print('overall black box acc: {}'.format(np.mean(overall_acc)))

    else:
        model = Model_crop(input_images, input_label)
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
        saver.restore(sess,'noatt_ckpt')
        mnist_adversary = np.load('mnist_adversary_noatt.npy').item()
        adv_x, adv_y = mnist_adversary['x'], mnist_adversary['y']
        overall_acc = []
        for itr in tqdm(range(300)):
            x_batch_test_adv, y_batch_test = adv_x[itr*batch_size:(itr+1)*batch_size], adv_y[itr*batch_size:(itr+1)*batch_size]
            adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                              input_label: y_batch_test}
            y_pred, test_loss_i = sess.run([model.y_pred, model.xent], feed_dict=adv_dict_test)
            counts = np.asarray([np.argmax(np.bincount(y_pred[:, i])) for i in range(batch_size)])
            test_acc_i = np.mean(counts == adv_dict_test[input_label])
            overall_acc += [test_acc_i]
            print("iter: {}, test_acc:{}  test_loss:{} ".format(itr, test_acc_i, test_loss_i))
        print('overall black box acc: {}'.format(np.mean(overall_acc)))
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
