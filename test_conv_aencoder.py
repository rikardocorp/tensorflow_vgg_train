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
    from vgg import conv_autoencoder as CAE
    from datasetTools import Dataset_csv
else:
    from tensorflow_vgg_train import utils
    from tensorflow_vgg_train.vgg import conv_autoencoder as CAE
    from tensorflow_vgg_train.datasetTools import Dataset_csv


# GLOBAL VARIABLES
path = 'features/'
path_data_train = [path+'Train_SNC4_relu6.csv']
path_data_test = [path+'Test_SNC4_relu6.csv']

path_load_weight = 'weight/BorrarAE_1.npy'
path_save_weight = 'weight/wcae_1.npy'

mini_batch_train = 34
mini_batch_test = 50
epoch = 40
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
        cost, layer = sess_test.run([CAEncode.cost, CAEncode.net['encode_1']], feed_dict={x_batch: x_})

        # save output of a layer
        # utils.save_layer_output(layer, label, name='Train_AE1_fc1')
        # print(layer.shape)

        cost_total = cost_total + cost
        objData.next_batch_test()

    return cost_total


# Funcion, fase de entrenamiento
def train_model(sess_train, objData, objDatatest):

    print('\n# PHASE: Training model')
    for ep in range(epoch):

        for i in range(objData.total_batchs):

            batch, _ = objData.generate_batch()
            sess_train.run(CAEncode.train, feed_dict={x_batch: batch})
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

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])
        CAEncode = CAE.CAEncoder(path_load_weight, learning_rate=learning_rate, noise=noise_level)
        CAEncode.build(input_batch=x_batch, n_filters=[1, 10, 10, 10], corruption=False)
        sess.run(tf.global_variables_initializer())

        print('Original Cost: ', test_model(sess, data_test))
        # train_model(sess_train=sess, objData=data_train, objDatatest=data_test)
        # test_model(sess_test=sess, objData=data_test)

        # SAVE WEIGHTs
        if path_save_weight is not None:
            CAEncode.save_npy(sess, path_save_weight)

        # PlOT RECONSTRUCTIONS
        n_examples = 5
        test_xs, _ = data_train.generate_batch()
        recon = sess.run(CAEncode.y, feed_dict={x_batch: test_xs})
        fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (64, 64)))
            axs[1][example_i].imshow(np.reshape(recon[example_i, :], (64, 64)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()

        # -----------------------------------------------------

        # import tensorflow.examples.tutorials.mnist.input_data as input_data
        #
        # path_load_weight = None
        # path_save_weight = 'weight/MNIST_AE3.npy'
        #
        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # trainX, testX = mnist.train, mnist.test
        # mean_img = np.mean(mnist.train.images, axis=0)
        # x_batch = tf.placeholder(tf.float32, [None, 784])
        #
        # CAEncode = CAE.CAEncoder(npy_path=path_load_weight, learning_rate=learning_rate, noise=noise_level)
        # CAEncode.build(input_batch=x_batch, corruption=False)
        # sess.run(tf.global_variables_initializer())
        #
        # # Fit all training data
        # print('Original: ', sess.run(CAEncode.cost, feed_dict={x_batch: mnist.test.images}))
        # batch_size = 20
        # n_epochs = 10
        # for epoch_i in range(n_epochs):
        #     for batch_i in range(mnist.train.num_examples // batch_size):
        #         batch_xs, _ = trainX.next_batch(batch_size)
        #         sess.run(CAEncode.train, feed_dict={x_batch: batch_xs})
        #
        #     print(epoch_i, sess.run(CAEncode.cost, feed_dict={x_batch: mnist.test.images}))
        #
        # CAEncode.save_npy(sess, path_save_weight)
        # # Plot example reconstructions
        # n_examples = 10
        # test_xs, _ = mnist.test.next_batch(n_examples)
        #
        # recon = sess.run(CAEncode.y, feed_dict={x_batch: test_xs})
        # fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        # for example_i in range(n_examples):
        #     axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (28, 28)))
        #     axs[1][example_i].imshow(np.reshape(recon[example_i, :], (28, 28)))
        #
        # fig.show()
        # plt.draw()
        # plt.waitforbuttonpress()

        # -----------------------------------------------------


