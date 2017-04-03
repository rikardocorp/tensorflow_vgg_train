import os
import numpy as np
import pandas as pd
import utils

class Dataset:

    def __init__(self, path_data='', path_dir_images='', minibatch=25):

        assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data
        data = pd.read_csv(path_data, header=None)
        self.path_data = path_data
        self.dir_images = path_dir_images
        self.images = data[0]
        self.labels = data[2]
        self.minibatch = minibatch
        self.total_images = len(data[0])
        self.start = 0
        self.end = minibatch
        self.batch = []
        assert (self.total_images / self.minibatch).is_integer(), 'El minibatch debe ser multiplo del total de datos de entrada.'
        self.shuffler()

    def generate_batch(self):

        start = self.start
        end = self.end
        batch_list = []
        label_list = []

        for i in range(start, end):
            # print(self.images[i], i)
            img = utils.load_image(self.dir_images + self.images[i] + '.jpg')[:, :, :3]
            batch_list.append(img.reshape((1, 224, 224, 3)))
            label_list.append(self.labels[i])

        return np.concatenate([block for block in batch_list], 0), label_list

    # def convert_data(self):
    #
    #     total = self.total_images
    #     item = []
    #     for i in range(1):
    #         # print(self.images[i], i)
    #         img = utils.load_image(self.dir_images + self.images[i] + '.jpg')[:, :, :3]
    #         # img = img.reshape((1, 224, 224, 3))
    #         img = img.reshape((150528))
    #         # item = json.dumps({"x": 0, "y": self.labels[i]}, sort_keys=True)
    #         print(img)
    #
    #         img = img.reshape((1, 224, 224, 3))

    def next_batch(self):

        if (self.end / self.total_images) == 1:
            self.start = 0
            self.end = self.minibatch
            self.shuffler()
        else:
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch

    def prev_batch(self):

        if self.start == 0:
            self.start = self.total_images - self.minibatch
            self.end = self.total_images
        else:
            self.start = self.start - self.minibatch
            self.end = self.end - self.minibatch

    def shuffler(self):
        df = pd.read_csv(self.path_data, header=None)
        df = df.reindex(np.random.permutation(df.index))
        df = pd.DataFrame(df).reset_index(drop=True)
        self.images = df[0]
        self.labels = df[2]