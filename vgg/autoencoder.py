import tensorflow as tf
import time
import numpy as np


class AEncoder:

    def __init__(self, npy_path=None, trainable=True, learning_rate=0.001, dropout=0.5, noise=0.3):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.noise = noise

    def build(self, input_batch, input_mask, train_mode=None, sess=None):
        self.sess = sess
        start_time = time.time()
        innBatch = input_batch * input_mask

        self.fc1 = self.fc_layer_sigm(innBatch, 4096, 2048, "fc1")
        self.prob = self.fc_layer_sigm(self.fc1, 2048, 4096, "prob", decode_w='fc1')

        # self.fc1 = self.fc_layer_sigm(innBatch, 784, 500, "fc1")
        # self.prob = self.fc_layer_sigm(self.fc1, 500, 784, "prob", decode_w='fc1')

        # Calculamos el error
        self.cost = tf.reduce_sum(tf.pow(input_batch - self.prob, 2))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def fc_layer_sigm(self, bottom, in_size, out_size, name, decode_w=None):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, type=2, decode_w=decode_w)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
            return fc

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name, type=1, decode_w=None):

        if type == 1 and decode_w is None:
            initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

            initial_value = tf.truncated_normal([out_size], .0, .001)
            biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        elif type == 2 and decode_w is None:
            w_init_max = 4 * np.sqrt(6. / (in_size + out_size))
            initial_value = tf.random_uniform([in_size, out_size],
                                              minval=-w_init_max,
                                              maxval=w_init_max)
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights")
            initial_value = tf.zeros([out_size])
            biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        if decode_w is not None:
            initial_value = tf.transpose(self.var_dict[(decode_w, 0)])
            # w_init_max = 4 * np.sqrt(6. / (in_size + out_size))
            # initial_value = tf.random_uniform([in_size, out_size],
            #                                   minval=-w_init_max,
            #                                   maxval=w_init_max)

            # weights = initial_value
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights", is_variable=True)
            if type == 1:
                initial_value = tf.truncated_normal([out_size], .0, .001)
                biases = self.get_var_fc(initial_value, name, 1, name + "_biases")
            elif type == 2:
                initial_value = tf.zeros([out_size])
                biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var_fc(self, initial_value, name, idx, var_name, is_variable=False):

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            if is_variable is False or self.data_dict is not None:
                var = tf.Variable(value, name=var_name)
            else:
                var = value
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var
    
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("File saved", npy_path)
        return npy_path
