"""
Expert tester for the vgg19_trainable
"""
import time
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix

switch_server = False

if switch_server is True:
    import utils
    from vgg import mlperceptron as mlp
    from datasetTools import Dataset_csv
else:
    from tensorflow_vgg_train import utils
    from tensorflow_vgg_train.vgg import mlperceptron as mlp
    from tensorflow_vgg_train.datasetTools import Dataset_csv


# GLOBAL VARIABLES
path = 'features/'
path_data_train = [path+'output_Train_SNC4_relu6.csv']
path_data_test  = [path+'output_Test_SNC4_relu6.csv']
path_list_labels = '../data/ISB2016/synset_skin.txt'
path_load_weight = 'weight/saveMlpB_4.npy'
path_save_weight = 'weight/saveMlpB_4.npy'
mini_batch_train = 20
mini_batch_test = 30
epoch = 40
num_class = 2
learning_rate = 0.001

print(path_data_train)
print(path_data_test)

# VALIDATE INPUT DATA
# assert (total_images / mini_batch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada'
assert os.path.exists(path), ('No existe el directorio de datos ' + path)
assert os.path.exists(path_data_train[0]), ('No existe el archivo con los datos de entrenamiento ' + path_data_train)
assert os.path.exists(path_data_test[0]), ('No existe el archivo con los datos de pruebas ' + path_data_test)
# assert os.path.exists(path_load_weight), 'No existe el archivo con pesos ' + path_load_weight


# Función, fase de test
def test_model(sess_test, objData):

    # sess_test : session en tensorflow
    # objData   : datos de test
    total = objData.total_inputs
    mbach = objData.minibatch

    if ((total/mbach) - int(total/mbach)) > 0:
        itertotal = int(total/mbach) + 1
    else:
        itertotal = int(total/mbach)

    count_success = 0
    count_by_class = np.zeros([num_class, num_class])
    prob_predicted = []

    # Iteraciones por Batch, en cada iteracion la session de tensorflow procesa los 'n' datos de entrada
    # donde 'n' es el 'mini_batch_test'
    print('\n# PHASE: Test classification')
    for i in range(itertotal):

        # Generamos el batch y sus respectivas etiquetas
        # el batch generado contiene las 'n' primeras imagenes
        batch, label = objData.generate_batch()

        # ejecutamos el grafo de tensorflow y almacenamos el vector de la ultima capa
        prob = sess_test.run(MLP.prob, feed_dict={mlp_batch: batch, train_mode: False})

        # save output of a layer
        # utils.save_layer_output(layer, label, name='Train_SNC4_relu6')

        # Acumulamos la presicion de cada iteracion, para despues hacer un promedio
        count, count_by_class, prob_predicted = utils.print_accuracy(label, prob, matrix_confusion=count_by_class, predicted=prob_predicted)
        count_success = count_success + count

        # hacemos que el batch apunte a los siguiente grupo de imagenes de tamaño 'n'
        objData.next_batch_test()

    # promediamos la presicion total
    accuracy_final = count_success/total
    print('\n# STATUS: Confusion Matrix')
    print(count_by_class)
    print('    Success total: ', str(count_success))
    print('    Accuracy total: ', str(accuracy_final))

    # a = objData.labels.tolist()
    # b = prob_predicted
    # cm = confusion_matrix(a, b)
    return accuracy_final


# Funcion, fase de entrenamiento
def train_model(sess_train, objData):

    # sess_train : session en tensorflow
    # objData   : datos de entrenamiento

    print('\n# PHASE: Training model')
    for ep in range(epoch):
        print('\n     Epoch:', ep)
        t0 = time.time()
        for i in range(objData.total_batchs):
            batch, label = objData.generate_batch()

            # Generate the 'one hot' or labels
            label = tf.one_hot([li for li in label], on_value=1, off_value=0, depth=num_class)
            label = list(sess_train.run(label))
            # Run training
            t_start = time.time()
            sess_train.run(MLP.train, feed_dict={mlp_batch: batch, mlp_label: label, train_mode: True})
            t_end = time.time()
            # Next slice batch
            objData.next_batch()
            print("        > Minibatch: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))

        t1 = time.time()
        # print("        Batch size: %d" % mini_batch_train)
        print("        Time epoch: %7.3f seg." % (t1 - t0))
        print("        Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))


if __name__ == '__main__':

    # GENERATE DATA
    # Datos de media y valor maximo
    data_normal = Dataset_csv(path_data=[path_data_train[0], path_data_test[0]], restrict=False, random=False)
    Damax = data_normal.amax
    print(Damax)
    # Damax = 1
    # Load data train
    data_train = Dataset_csv(path_data=path_data_train, minibatch=mini_batch_train, max_value=Damax)
    # Load data test
    data_test = Dataset_csv(path_data=path_data_test, minibatch=mini_batch_test, max_value=Damax, restrict=False, random=False)
    accuracy = 0

    with tf.Session() as sess:

        # DEFINE MODEL
        mlp_batch = tf.placeholder(tf.float32, [None, 4096])
        mlp_label = tf.placeholder(tf.float32, [None, num_class])
        train_mode = tf.placeholder(tf.bool)

        MLP = mlp.MLPerceptron(path_load_weight, learning_rate=learning_rate, size_layer_fc=2048)
        MLP.build(mlp_batch, mlp_label, train_mode)

        sess.run(tf.global_variables_initializer())
        test_model(sess_test=sess, objData=data_test)
        # train_model(sess_train=sess, objData=data_train)
        # accuracy = test_model(sess_test=sess, objData=data_test)
        # #
        # # SAVE LOG: Genera un registro en el archivo log-server.txt
        # utils.write_log(total_data=data_train.total_inputs,
        #                 epoch=epoch,
        #                 m_batch=mini_batch_train,
        #                 l_rate=learning_rate,
        #                 accuracy=accuracy,
        #                 file_npy=path_load_weight)
        #
        # # SAVE WEIGHTs
        # MLP.save_npy(sess, path_save_weight)
