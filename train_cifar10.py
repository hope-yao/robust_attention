import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model_cifar10 import *
from pgd_multiGPU import *
from viz2 import *
import os
import numpy as np
from utils import creat_dir
from tqdm import tqdm
from cifar10_loader import load
slim = tf.contrib.slim

def main(cfg):
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    input_images = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, 3))
    input_label = tf.placeholder(tf.int64,shape=(batch_size))

    # build classifier
    # model = Model_att(input_images, input_label, cfg)
    model = Model_Madry(input_images, input_label, cfg, mode='train')

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

    hist = {'train_acc': [],
            'train_adv_acc': [],
            'test_acc': [],
            'test_adv_acc': [],
            'train_loss': [],
            'test_loss': [],
            'train_adv_loss': [],
            'test_adv_loss': []}
    train_iters=500000
    trainX, trainY, testX, testY = load(one_hot=False, cv=-1)
    num_batch_per_epoch = trainY.shape[0] // batch_size
    for itr in tqdm(range(train_iters)):
        ii = itr%num_batch_per_epoch
        x_batch_train, y_batch_train = trainX[ii*batch_size:(ii+1)*batch_size], trainY[ii*batch_size:(ii+1)*batch_size]
        nat_dict_train = {input_images: x_batch_train.reshape(batch_size, img_size, img_size, 3),
                          input_label: y_batch_train}
        x_batch_train_adv = get_PGD(sess, model.adv_grad, nat_dict_train, input_images, epsilon=8./255, a=2./255, k=8)
        adv_dict_train = {input_images: x_batch_train_adv.reshape(batch_size, img_size, img_size, 3),
                          input_label: y_batch_train}
        sess.run(model.train_op, feed_dict=adv_dict_train)

        if itr % 100 == 0:
            train_acc_i, train_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_train)
            jj = np.random.randint(len(testY)// batch_size)
            x_batch_test, y_batch_test = testX[jj*batch_size:(jj+1)*batch_size], testY[jj*batch_size:(jj+1)*batch_size]
            nat_dict_test = {input_images: x_batch_test.reshape(batch_size, img_size, img_size, 3),
                              input_label: y_batch_test}
            test_acc_i, test_loss_i = sess.run([model.accuracy, model.xent], feed_dict=nat_dict_test)
            print("iter: {}, train_acc:{}  test_acc:{} train_loss:{}  test_loss:{} "
                  .format(itr, train_acc_i, test_acc_i, train_loss_i, test_loss_i))

            x_batch_train_adv = get_PGD(sess, model.adv_grad, nat_dict_train, input_images, epsilon=8./255, a=2./255, k=20)
            adv_dict_train = {input_images: x_batch_train_adv.reshape(batch_size, img_size, img_size, 3),
                              input_label: y_batch_train}
            train_adv_acc_i, train_adv_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_train)
            x_batch_test_adv = get_PGD(sess, model.adv_grad, nat_dict_test, input_images, epsilon=8./255, a=2./255, k=20)
            adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 3),
                              input_label: y_batch_test}
            test_adv_acc_i, test_adv_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_test)
            print("iter: {}, train_adv_acc:{}  test_adv_acc:{} train_adv_loss:{}  test_adv_loss:{} "
                .format(itr, train_adv_acc_i, test_adv_acc_i, train_adv_loss_i, test_adv_loss_i))
            hist['train_acc'] += [train_acc_i]
            hist['train_adv_acc'] += [train_adv_acc_i]
            hist['test_acc'] += [test_acc_i]
            hist['test_adv_acc'] += [test_adv_acc_i]
            hist['train_loss'] += [train_loss_i]
            hist['test_loss'] += [test_loss_i]
            hist['train_adv_loss'] += [train_adv_loss_i]
            hist['test_adv_loss'] += [test_adv_loss_i]
            np.save('hist',hist)
            saver.save(sess, 'ckpt')
    print('done')


if __name__ == "__main__":


    cfg = {'batch_size': 4,
           'img_dim': 2,
           'img_size': 32,
           'num_glimpse': 5,
           'glimpse_size': 20,
           'lr': 1e-4,
           'nGPU': 1
           }
    main(cfg)