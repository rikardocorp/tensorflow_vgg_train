import tensorflow as tf
import time
import numpy as np


class AEncoder:

    def __init__(self, npy_path=None, trainable=True, dropout=0.5, load_weight_ae=False):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.load_weight_fc = load_weight_ae

    def build(self, input, train_mode=None):

        start_time = time.time()
        print("build model started")

        self.fc1 = self.fc_layer(input, 4096, 1024, "fc1")
        self.relu1 = tf.nn.relu(self.fc1)

        # DROPOUT
        if train_mode is not None:
            # train_mode: True[train] -> dropout activate | False[test] -> dropout deactivate
            self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        elif self.trainable:
            # train_mode: None -> Default [test or classification]
            self.relu1 = tf.nn.dropout(self.relu1, self.dropout)

        self.fc2 = self.fc_layer(self.relu1, 1024, 4096, "fc8")

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var_fc(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict and self.load_weight_fc is True:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var