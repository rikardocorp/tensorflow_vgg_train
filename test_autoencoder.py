"""
Expert tester for the vgg19_trainable
"""
import time
import tensorflow as tf
import os
import numpy as np

switch_server = False

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
path_data_train = [path+'outputrain.csv']
path_data_test = [path+'outputest.csv']

mini_batch_train = 5
mini_batch_test = 4
epoch = 200
learning_rate = 0.001


# Función, fase de test
def test_model(sess_test, objData):

    total = objData.total_inputs
    mbach = objData.minibatch

    if ((total/mbach) - int(total/mbach)) > 0:
        itertotal = int(total/mbach) + 1
    else:
        itertotal = int(total/mbach)

    count_success = 0
    print('\n# PHASE: Test classification')
    for i in range(itertotal):

        batch, _ = objData.generate_batch()
        prob, error = sess_test.run([AEncode.prob, AEncode.errorDecode], feed_dict={x_batch: batch, train_mode: False})

        # print(prob.shape, batch.shape)
        # print(error.shape)
        # print(prob)
        # print()
        # print(batch)
        # print()
        print(error)

        objData.next_batch_test()
        print('-------')

    # promediamos la presicion total
    # accuracy_final = count_success/total
    # print('    Success total: ', str(count_success))
    # print('    Accuracy total: ', str(accuracy_final))
    # return accuracy_final


# Funcion, fase de entrenamiento
def train_model(sess_train, objData):

    total = objData.total_inputs
    cost = tf.reduce_mean((AEncode.prob - x_batch) ** 2)
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    print('\n# PHASE: Training model')
    for ep in range(epoch):
        print('\n     Epoch:', ep)
        t0 = time.time()
        for i in range(objData.total_batchs):

            batch, label = objData.generate_batch()
            # Run training
            t_start = time.time()
            sess_train.run(train, feed_dict={x_batch: batch, train_mode: True})
            t_end = time.time()
            # Next slice batch
            objData.next_batch()
            print("        > Minibatch: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))

        t1 = time.time()
        # print("        Batch size: %d" % mini_batch_train)
        print("        Time epoch: %7.3f seg." % (t1 - t0))
        print("        Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))


if __name__ == '__main__':

    # Datos de media y valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train[0], path_data_test[0]], restrict=False, random=False)
    Damax = data_normal.amax
    Dmean = data_normal.media_mean

    # Load data train
    data_train = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train)
    data_train.amax = Damax
    data_train.media_mean = Dmean

    # Load data test
    data_test = Dataset_csv(path_data=path_data_test, minibatch=mini_batch_test, restrict=False, random=False)
    data_test.amax = Damax
    data_test.media_mean = Dmean

    with tf.Session() as sess:

        x_batch = tf.placeholder(tf.float32, [None, 4096])
        y_batch = tf.placeholder(tf.float32, [None, 4096])
        train_mode = tf.placeholder(tf.bool)

        AEncode = AE.AEncoder(learning_rate=learning_rate)
        AEncode.build(input_batch=x_batch, train_mode=train_mode)

        # sess.run(tf.global_variables_initializer())
        # test_model(sess_test=sess, objData=data_test)
        # train_model(sess_train=sess, objData=data_train)
        # test_model(sess_test=sess, objData=data_test)
