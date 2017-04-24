"""
AUTOR :
    Ricardo Coronado Pérez
    
DESCRIPCION:
    Esta clase a sido diseñada para manipular un conjunto de datos de imagenes,
    para ques esto funcione es necesario contar con un directorio de imagenes y 
    un archivo csv contenga en la:
    
        - primera columna el nombre del archivo (imagen sin extension)
        - segunda columna la clase a la que pertenece
        - tercera columna la clase enumerada en un valor entero (ej. si contamos con 5 clases, 
          tendremos las clases enumeradas del 0 al 4)
    
    aunque el objetivo de esta clase es para cargar imagenes, puede modificarse el codigo para que pueda trabajar 
    con otro tipo de datos
    
FUNCIONES:
    generar bach : genera el primer conjunto de imagenes y sus etiquetas
    next batch : salta los indices de los punteros al siguiente conjunto de imagenes
    prev batch : salta los indices de los punteros al conjunto de datos anterior
    shuffler : re-ordena toda la data
    
"""

import os
import numpy as np
import pandas as pd
import utils
# from tensorflow_vgg_train import utils

# ----------------------- #
# MANAGE DATASET FROM NPY #
# ----------------------- #

class Dataset:

    """
        path_data: ruta del archivo csv que almacena el nombre de la imagen y su etiqueta
        path_dir_images: directorio donde se encuentran las imagenes
        minibatch: es el tamaño del conjunto de imagenes 
        cols: son los indices de las columnas del 'nombre de la imagen' y el 'label o clase'
    """

    def __init__(self, path_data='', path_dir_images='', minibatch=25, cols=[], restrict=True, random=True):

        assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data

        self.path_data = path_data
        self.dir_images = path_dir_images
        self.minibatch = minibatch
        self.cols = cols

        # leemos el archivo csv y guardamos las columnas 0 y 2 (nombre de imagen y etiqueta respectivamente)
        data = pd.read_csv(path_data, header=None)
        self.images = data[cols[0]]
        self.labels = data[cols[1]]
        self.total_images = len(data[cols[0]])

        # inicializamos los punteros de la data
        self.start = 0
        self.end = minibatch

        if restrict is True:
            assert (self.total_images / self.minibatch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada.'

        self.total_batchs = int(self.total_images / self.minibatch)

        # Realizamos un reordenamiento por defecto
        if random is True:
            self.shuffler()

    #
    # Generamos el batch en la posicion actual donde se encuentras los punteros self.start y self.end
    def generate_batch(self):

        start = self.start
        end = self.end
        batch_list = []
        label_list = []

        # cargamos las imagenes y estas son tratadas para darles el tamaño requerido
        for i in range(start, end):
            # print(self.images[i], i)
            img = utils.load_image(self.dir_images + self.images[i] + '.jpg')[:, :, :3]
            batch_list.append(img.reshape((1, 224, 224, 3)))
            label_list.append(self.labels[i])

        # concatena cada elemento del batch_list dando como resultado una matriz con la forma (n, 224, 224, 3)
        # retorna el batch_list concatenado y el label_list con n items, una etiqueta para cada imagen
        return np.concatenate([block for block in batch_list], 0), label_list

    #
    # Recorre la lista de imagenes de atras hacia adelante
    def next_batch(self):

        # es positivo cuando se llega al ultimo batch
        if (self.end / self.total_images) == 1:
            # inicializa los indices y reordena la posicion de las imagenes
            self.start = 0
            self.end = self.minibatch
            self.shuffler()
        else:
            # hace que los indices apunte a las siguientes imagenes
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch

    #
    # Recorre de manera especial la lista para la fase de entrenamiento, cuando no existe la restriccion del minibatch
    def next_batch_test(self):

        dif = self.total_images - self.end
        dif_div = dif/self.minibatch

        if dif_div >= 1:
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch
        elif dif_div == 0:
            self.start = 0
            self.end = self.minibatch
        elif dif_div < 1:
            self.start = self.start + self.minibatch
            self.end = self.total_images

    #
    # Recorre la lista de imagenes de adelante hacia atras
    def prev_batch(self):

        # es positivo cuando el indice llega al primer batch
        if self.start == 0:
            # inicializa los indices para que apunten al bach final
            self.start = self.total_images - self.minibatch
            self.end = self.total_images
            self.shuffler()
        else:
            # actuliza los indices
            self.start = self.start - self.minibatch
            self.end = self.end - self.minibatch

    #
    # Reordena la lista de imagenes, simmula la aleatoridad en la eleccion de batchs
    def shuffler(self):
        cols = self.cols
        df = pd.read_csv(self.path_data, header=None)
        df = df.reindex(np.random.permutation(df.index))
        df = pd.DataFrame(df).reset_index(drop=True)
        self.images = df[cols[0]]
        self.labels = df[cols[1]]


# ----------------------- #
# MANAGE DATASET FROM CSV #
# ----------------------- #

class Dataset_csv:

    def __init__(self, path_data=[], minibatch=25, restrict=True, random=True, max_value=None, media_mean=None):

        assert len(path_data) > 0, 'No se ingresaron archivos con los datos de entrada.'

        self.path_data = path_data
        self.minibatch = minibatch

        # leemos el archivo csv y guardamos las columnas 0 y 2 (nombre de imagen y etiqueta respectivamente)
        df = []
        for file in path_data:
            assert os.path.exists(file), 'No existe el archivo con los datos de entrada ' + file
            aux = pd.read_csv(file, header=None)
            df.append(aux)

        self.data = pd.concat(df).reset_index(drop=True)

        self.inputs = self.data.iloc[:, :-1]

        if max_value is None:
            self.amax = self.inputs.max().max()
        else:
            self.amax = max_value

        self.inputs = self.inputs / self.amax
        self.labels = self.data.iloc[:, -1:].astype(int)
        self.total_inputs = len(self.inputs)

        if media_mean is None:
            self.media_mean = np.mean(self.inputs, axis=0)
        else:
            self.media_mean = media_mean

        # inicializamos los punteros de la data
        self.start = 0
        self.end = minibatch

        if restrict is True:
            assert (self.total_inputs / self.minibatch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada.'

        self.total_batchs = int(self.total_inputs / self.minibatch)

        # Realizamos un reordenamiento por defecto
        if random is True:
            self.shuffler()

    # Normalizamos la data entre 0 y 1 en base al valor maximo
    def normalization(self, max=1.0):
        self.amax = max
        self.inputs = self.data.iloc[:, :-1]
        self.inputs = self.inputs / self.amax
        self.media_mean = np.mean(self.inputs, axis=0)
        print('Dataset normalizado.')

    #
    # Generamos el batch en la posicion actual donde se encuentras los punteros self.start y self.end
    def generate_batch(self):

        start = self.start
        end = self.end
        batch_list = []
        label_list = []

        countBatch = end - start
        countCols  = len(self.inputs.columns)

        for i in range(start, end):
            # print(i)
            orig = self.inputs.iloc[i, :].values
            mean = self.media_mean
            batch_list.append(orig)
            label_list.append(self.labels.iloc[i, :].values[0])

        return np.reshape(batch_list, (countBatch, countCols)), label_list

    #
    # Recorre la lista de imagenes de atras hacia adelante
    def next_batch(self):

        # es positivo cuando se llega al ultimo batch
        if (self.end / self.total_inputs) == 1:
            # inicializa los indices y reordena la posicion de las imagenes
            self.start = 0
            self.end = self.minibatch
            self.shuffler()
        else:
            # hace que los indices apunte a las siguientes imagenes
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch

    #
    # Recorre la lista de imagenes de adelante hacia atras
    def prev_batch(self):

        # es positivo cuando el indice llega al primer batch
        if self.start == 0:
            # inicializa los indices para que apunten al bach final
            self.start = self.total_inputs - self.minibatch
            self.end = self.total_inputs
            # self.shuffler()
        else:
            # actuliza los indices
            self.start = self.start - self.minibatch
            self.end = self.end - self.minibatch

    #
    # Recorre de manera especial la lista para la fase de entrenamiento, cuando no existe la restriccion del minibatch
    def next_batch_test(self):

        dif = self.total_inputs - self.end
        dif_div = dif/self.minibatch

        if dif_div >= 1:
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch
        elif dif_div == 0:
            self.start = 0
            self.end = self.minibatch
        elif dif_div < 1:
            self.start = self.start + self.minibatch
            self.end = self.total_inputs

    #
    # Reordena la lista de imagenes, simmula la aleatoridad en la eleccion de batchs
    def shuffler(self):

        df = self.data
        df = df.reindex(np.random.permutation(df.index))
        df = pd.DataFrame(df).reset_index(drop=True)

        self.inputs = df.iloc[:, :-1] / self.amax
        self.labels = df.iloc[:, -1:].astype(int)
