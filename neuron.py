# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import tensorflow as tf


CIFAR_DIR = 'data/cifar-10-batches-py'
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    """
    read data from data file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CiferData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            for item, label in zip(data, labels):
                if label in [0, 1]:
                    all_data.append(item)
                    all_labels.append(label)
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """
        return batch_size examples as a batch.
        :param batch_size:
        :return:
        """
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


x = tf.placeholder(tf.float32, [None, 3072])
# [None]
y = tf.placeholder(tf.int64, [None])

# [3072, 1]
w = tf.get_variable('w', [x.get_shape()[-1], 1], initializer=tf.random_normal_initializer(0, 1))
# [1, ]
b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
# [None, 3072] * [3072, 1] = [None, 1]
y_ = tf.matmul(x, w) + b
# [None, 1]
p_y_1 = tf.nn.sigmoid(y_)
# [None, 1]
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
loss = tf.reduce_mean(tf.square(y_reshaped - p_y_1))
predict = p_y_1 > 0.5
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()



