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


class Dataset:

    """
        path_data: ruta del archivo csv que almacena el nombre de la imagen y su etiqueta
        path_dir_images: directorio donde se encuentran las imagenes
        minibatch: es el tamaño del conjunto de imagenes 
    """

    def __init__(self, path_data='', path_dir_images='', minibatch=25):

        assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data

        self.path_data = path_data
        self.dir_images = path_dir_images
        self.minibatch = minibatch

        # leemos el archivo csv y guardamos las columnas 0 y 2 (nombre de imagen y etiqueta respectivamente)
        data = pd.read_csv(path_data, header=None)
        self.images = data[0]
        self.labels = data[2]
        self.total_images = len(data[0])

        # inicializamos los punteros de la data
        self.start = 0
        self.end = minibatch
        self.batch = []
        assert (self.total_images / self.minibatch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada.'

        # Realizamos un reordenamiento por defecto
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
        df = pd.read_csv(self.path_data, header=None)
        df = df.reindex(np.random.permutation(df.index))
        df = pd.DataFrame(df).reset_index(drop=True)
        self.images = df[0]
        self.labels = df[2]
