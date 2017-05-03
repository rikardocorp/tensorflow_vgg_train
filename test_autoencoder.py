"""
Expert tester for the vgg19_trainable
"""
import time
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

switch_server = True

if switch_server is True:
    import utils
    from vgg import autoencoder as AE
    from datasetTools import Dataset_csv
else:
    from tensorflow_vgg_train import utils
    from tensorflow_vgg_train.vgg import autoencoder as AE
    from tensorflow_vgg_train.datasetTools import Dataset_csv


# GLOBAL VARIABLES
path = 'features/'
path_data_train = [path+'Train_SNC4_relu6.csv']
path_data_test = [path+'Test_SNC4_relu6.csv']

path_load_weight = None
path_save_weight = 'weight/saveAE_2.npy'

mini_batch_train = 20
mini_batch_test = 25
epoch = 4
learning_rate = 0.0001
noise_level = 0.0


# Función, fase de test
def test_model(sess_test, objData):

    total = objData.total_inputs
    mbach = objData.minibatch
    if ((total/mbach) - int(total/mbach)) > 0:
        itertotal = int(total/mbach) + 1
    else:
        itertotal = int(total/mbach)

    cost_total = 0
    noise = 0
    for i in range(itertotal):

        x_, label = objData.generate_batch()
        mask_np = np.random.binomial(1, 1 - noise, x_.shape)
        cost, layer = sess_test.run([AEncode.cost, AEncode.net['encodeFC_1']], feed_dict={x_batch: x_, mask: mask_np})

        # save output of a layer
        # utils.save_layer_output(layer, label, name='Train_AE1_fc1')

        cost_total = cost_total + cost
        objData.next_batch_test()

    return cost_total


# Funcion, fase de entrenamiento
def train_model(sess_train, objData, objDatatest):

    print('\n# PHASE: Training model')
    for ep in range(epoch):

        for i in range(objData.total_batchs):

            batch, _ = objData.generate_batch()
            mask_np = np.random.binomial(1, 1 - noise_level, batch.shape)
            sess_train.run(AEncode.train, feed_dict={x_batch: batch, mask: mask_np})
            objData.next_batch()

        cost_prom = test_model(sess_train, objDatatest)
        print('     Epoch', ep, ': ', cost_prom)


if __name__ == '__main__':

    # Datos de media y valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train[0], path_data_test[0]], restrict=False, random=False)
    Damax = data_normal.amax

    # Load data train
    data_train = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train, max_value=Damax)
    # Load data test
    data_test = Dataset_csv(path_data=path_data_test, minibatch=mini_batch_test, max_value=Damax, restrict=False, random=False)
    # data_test = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train, max_value=Damax, restrict=False, random=False)

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])
        mask = tf.placeholder(tf.float32, [None, 4096])
        # train_mode = tf.placeholder(tf.bool)

        AEncode = AE.AEncoder(path_load_weight, learning_rate=learning_rate, noise=noise_level)
        AEncode.build(input_batch=x_batch, input_mask=mask, l_hidden=[2048, 1024])

        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(sess, data_test))
        train_model(sess_train=sess, objData=data_train, objDatatest=data_test)
        # test_model(sess_test=sess, objData=data_test)

        # SAVE WEIGHTs
        # if path_save_weight is not None:
        #     AEncode.save_npy(sess, path_save_weight)

        # Plot example reconstructions
        n_examples = 5
        test_xs, _ = data_train.generate_batch()
        test_xs_norm = np.array(test_xs)
        mask_np = np.random.binomial(1, 1 - noise_level, test_xs.shape)

        recon = sess.run(AEncode.y, feed_dict={x_batch: test_xs_norm, mask: mask_np})
        fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (64, 64)))
            axs[1][example_i].imshow(
                np.reshape(recon[example_i, :], (64, 64)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()

        # -----------------------------------------------------

        # import tensorflow.examples.tutorials.mnist.input_data as input_data
        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
        #
        # x_batch = tf.placeholder(tf.float32, [None, 784])
        # mask = tf.placeholder(tf.float32, [None, 784])
        # # train_mode = tf.placeholder(tf.bool)
        #
        # AEncode = AE.AEncoder(npy_path=None, learning_rate=learning_rate, noise=noise_level)
        # AEncode.build(input_batch=x_batch, input_mask=mask, l_hidden=[500])
        #
        # # you need to initialize all variables
        # sess.run(tf.global_variables_initializer())
        #
        # test_xs = mnist.test.images
        # mask_np = np.random.binomial(1, 1 - noise_level, test_xs.shape)
        # print('Original: ', sess.run(AEncode.cost, feed_dict={x_batch: test_xs, mask: mask_np}))
        # total = len(trX)
        # epoch = 30
        # # total = 1500
        # for i in range(epoch):
        #     for start, end in zip(range(0, total, 128), range(128, total, 128)):
        #         input_ = trX[start:end]
        #         mask_np = np.random.binomial(1, 1 - noise_level, input_.shape)
        #         sess.run(AEncode.train, feed_dict={x_batch: input_, mask: mask_np})
        #
        #     mask_np = np.random.binomial(1, 1 - noise_level, teX.shape)
        #     print(i, sess.run(AEncode.cost, feed_dict={x_batch: teX, mask: mask_np}))
        #
        # AEncode.save_npy(sess, 'weight/MNIST_AE2.npy')
        # # %%
        # # Plot example reconstructions
        # n_examples = 10
        # test_xs, _ = mnist.test.next_batch(n_examples)
        # test_xs_norm = np.array(test_xs)
        # mask_np = np.random.binomial(1, 1 - noise_level, test_xs.shape)
        #
        # recon = sess.run(AEncode.y, feed_dict={x_batch: test_xs_norm, mask: mask_np})
        # fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        # for example_i in range(n_examples):
        #     axs[0][example_i].imshow(
        #         np.reshape(test_xs[example_i, :], (28, 28)))
        #     axs[1][example_i].imshow(
        #         np.reshape(recon[example_i, :], (28, 28)))
        # fig.show()
        # plt.draw()
        # plt.waitforbuttonpress()

        # -----------------------------------------------------


