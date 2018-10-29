import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model_mnist import Model_madry as Model
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

    # build classifier
    #model = Model(input_images, input_label, glimpse_size, num_glimpse)
    model = Model(input_images, input_label)
    attack = LinfPGDAttack(model, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func='xent')
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
    saver.restore(sess,'nat_noatt_ckpt')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    mnist_adversary = {'x':[], 'y':[]}
    for itr in tqdm(range(300)):
        x_batch_test, y_batch_test = mnist.test.next_batch(batch_size)
        x_batch_test_adv = attack.perturb(x_batch_test.reshape(batch_size, img_size, img_size, 1), y_batch_test, sess)
        adv_dict_test = {input_images: x_batch_test_adv.reshape(batch_size, img_size, img_size, 1),
                          input_label: y_batch_test}
        mnist_adversary['x'] += [x_batch_test_adv]
        mnist_adversary['y'] += [y_batch_test]
        test_acc_i, test_loss_i = sess.run([model.accuracy, model.xent], feed_dict=adv_dict_test)
        print("iter: {}, test_acc:{}  test_loss:{} ".format(itr, test_acc_i, test_loss_i))
    mnist_adversary['x'] = np.concatenate(mnist_adversary['x'],0)
    mnist_adversary['y'] = np.concatenate(mnist_adversary['y'],0)
    np.save('mnist_adversary',mnist_adversary)

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
