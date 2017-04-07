"""
Expert tester for the vgg19_trainable
"""
import time
import tensorflow as tf
import utils
import os

# importamos la arquitectura de la red
from vgg import vgg19_trainable_skin as vgg19

# importamos la clase Dataset para generar los batch
from datasetTools import Dataset

# GLOBAL VARIABLES
path = '../data/ISB2016/'
path_dir_image_train = path + "image_train/"
path_dir_image_test = path + "image_test/"
path_list_labels = path + 'synset_skin.txt'
path_load_weight = 'weight/vgg19.npy'
path_save_weight = 'weight/save_1.npy'

# VARIABLES MODEL
path_data_train = path + 'ISB_Train.csv'
path_data_test = path + 'ISB_Test.csv'
mini_batch_train = 15
mini_batch_test = 10
epoch = 2
num_class = 2
learning_rate = 0.01


# Variable para cargar los pesos de la capa fullConnect
# False : Cuando modificamos las capas fc para que el modelo clasifique un nuevo tipo de datos
# True  : Cuando ya contamos con un archivo de pesos entrenados .npy de la nueva red podemos cargarlos
# Nota, siempre que utilicemos inicialmente los pesos originales del archivo vgg19.npy debemos setear la variable en False
# ya que este archivo almacena los pesos de la red vgg original, al cargarlos en nuestra red ocurrira un error.
load_weight_fc = False


# VALIDATE INPUT DATA
# assert (total_images / mini_batch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada'
assert os.path.exists(path), 'No existe el directorio de datos ' + path
assert os.path.exists(path_list_labels), 'No existe el archivo con la lista de labels ' + path_list_labels
assert os.path.exists(path_data_train), 'No existe el archivo con los datos de entrenamiento ' + path_data_train
assert os.path.exists(path_data_test), 'No existe el archivo con los datos de pruebas ' + path_data_test
# assert os.path.exists(path_load_weight), 'No existe el archivo con pesos ' + path_load_weight


# Función, fase de test
def test_model(sess_test, objData):

    # sess_test : session en tensorflow
    # objData   : datos de test
    total = objData.total_images
    mbach = objData.minibatch

    if total/mbach > 1:
        itertotal = int(total/mbach) + 1
    else:
        itertotal = int(total/mbach)

    accuracy = 0

    # Iteraciones por Batch, en cada iteracion la session de tensorflow procesa los 'n' datos de entrada
    # donde 'n' es el 'mini_batch_test'
    print('\n# PHASE: Test classification')
    for i in range(itertotal):

        # Generamos el batch y sus respectivas etiquetas
        # el batch generado contiene las 'n' primeras imagenes
        batch, label = objData.generate_batch()

        # ejecutamos el grafo de tensorflow y almacenamos el vector de la ultima capa
        prob = sess_test.run(vgg.prob, feed_dict={vgg_batch: batch, train_mode: False})

        # Acumulamos la presicion de cada iteracion, para despues hacer un promedio
        accuracy = accuracy + utils.print_accuracy(label, prob)

        # hacemos que el batch apunte a los siguiente grupo de imagenes de tamaño 'n'
        objData.next_batch_test()

    # promediamos la presicion total
    accuracy_final = accuracy/itertotal
    print('    Accuracy total: ', str(accuracy_final))

    return accuracy_final


# Funcion, fase de entrenamiento
def train_model(sess_train, objData):

    # sess_test : session en tensorflow
    # objData   : datos de entrenamiento

    # Funciones para optimizar y entrenar el modelo
    total = objData.total_images
    # Promedio cuadrado : ((x - xi)^2)/n
    cost = tf.reduce_mean((vgg.prob - vgg_label) ** 2)
    # Optimizador del error, gradiente descendiente
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    print('\n# PHASE: Training model')
    t0 = time.time()
    for i in range(epoch):
        batch, label = objData.generate_batch()

        # Generate the 'one hot' or labels
        label = tf.one_hot([li for li in label], on_value=1, off_value=0, depth=num_class)
        label = list(sess_train.run(label))
        # Run training
        t_start = time.time()
        sess_train.run(train, feed_dict={vgg_batch: batch, vgg_label: label, train_mode: True})
        t_end = time.time()
        # Next slice batch
        objData.next_batch()
        print("    Iteration: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))

    t1 = time.time()
    print("    Batch size: %d" % mini_batch_train)
    print("    Iterations: %d" % epoch)
    print("    Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))


if __name__ == '__main__':

    # GENERATE DATA
    data_train = Dataset(path_data=path_data_train, path_dir_images=path_dir_image_train, minibatch=mini_batch_train, cols=[0,2])
    data_test = Dataset(path_data=path_data_test, path_dir_images=path_dir_image_test, minibatch=mini_batch_test, cols=[0,1], restrict=False)
    accuracy = 0

    with tf.Session() as sess:

        # DEFINE MODEL
        vgg_batch = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg_label = tf.placeholder(tf.float32, [None, num_class])
        train_mode = tf.placeholder(tf.bool)

        # Initialize of the model VGG19
        vgg = vgg19.Vgg19(path_load_weight, load_weight_fc=load_weight_fc)
        vgg.build(vgg_batch, train_mode)

        sess.run(tf.global_variables_initializer())
        test_model(sess_test=sess, objData=data_test)
        train_model(sess_train=sess, objData=data_train)
        accuracy = test_model(sess_test=sess, objData=data_test)

        # SAVE LOG: Genera un registro en el archivo log-server.txt
        utils.write_log(total_data=data_train.total_images,
                        epoch=epoch,
                        m_batch=mini_batch_train,
                        l_rate=learning_rate,
                        accuracy=accuracy,
                        file_npy=path_load_weight)

        # SAVE WEIGHTs
        vgg.save_npy(sess, path_save_weight)
