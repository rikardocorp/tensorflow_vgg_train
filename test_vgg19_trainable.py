"""
Simple tester for the vgg19_trainable
"""
import time

import numpy as np
import tensorflow as tf

import utils
from vgg import vgg19_trainable as vgg19

img1 = utils.load_image("./test_data/tiger.jpeg")[:, :, :3]    # Con [:, :, :3] podemos ahora cargar imagenes png y jpg
img2 = utils.load_image("./test_data/dog2.png")[:, :, :3]
img3 = utils.load_image("./test_data/lobo.png")[:, :, :3]

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2, batch3), 0)

with tf.Session() as sess:
    # with tf.device('/cpu:0'):
    #     sess = tf.Session()

    label = tf.one_hot([292, 182, 269], on_value=1, off_value=0, depth=1000)
    img_true_result = list(sess.run(label))

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./weight/vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())
    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    utils.print_prob_all(prob, './synset.txt', 0)
    print()

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # START - Training
    batch_size = len(batch)
    n = 2
    t0 = time.time()
    for i in range(n):
        tstart = time.time()
        sess.run(train, feed_dict={images: batch, true_out: img_true_result, train_mode: True})
        tend = time.time()
        print("Iteration: %d train on batch time: %7.3f ms." % (i, (tend - tstart) * 1000))
    t1 = time.time()

    print("Batch size: %d" % batch_size)
    print("Iterations: %d" % n)
    print("Time per iteration: %7.3f ms" % ((t1 - t0) * 1000 / n))
    # END - Train

    # test classification again, should have a higher probability about tiger
    prob, kernel = sess.run([vgg.prob, vgg.var_dict[('conv1_1', 0)]], feed_dict={images: batch, train_mode: False})
    print()
    utils.print_prob_all(prob, './synset.txt', 0)

    # test save
    # vgg.save_npy(sess, './weight/save-vgg19.npy')
