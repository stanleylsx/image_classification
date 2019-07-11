# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import tensorflow as tf


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
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
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
y = tf.placeholder(tf.int64, [None])
# [None]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32*32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])


# conv1: 神经元图,feature_map,输出图像
conv1_1 = tf.layers.conv2d(x_image,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv1_1'
                           )

conv1_2 = tf.layers.conv2d(conv1_1,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv1_2'
                           )

# 16*16
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool1'
                                   )

conv2_1 = tf.layers.conv2d(pooling1,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv2_1'
                           )

conv2_2 = tf.layers.conv2d(conv2_1,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv2_2'
                           )
# 8*8
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool2'
                                   )

conv3_1 = tf.layers.conv2d(pooling2,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv3_1'
                           )

conv3_2 = tf.layers.conv2d(conv3_1,
                           32,  # outout channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           name='conv3_2'
                           )

# 4*4*32
pooling3 = tf.layers.max_pooling2d(conv3_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool3'
                                   )
# 展平 [None, 4*4*32]
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_ -> softmax
# y -> one_hot
# loss = y_logy_

# indices
predict = tf.argmax(y_, 1)
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

if __name__ == '__main__':
    CIFAR_DIR = './data/cifar-10-batches-py'
    train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_{}'.format(i)) for i in range(1, 6)]
    test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
    train_data = CiferData(train_filenames, True)
    test_data = CiferData(test_filenames, False)
    init = tf.global_variables_initializer()
    batch_size = 20
    train_steps = 10000
    test_steps = 100
    # placeholders, metrics, train_op = create_model()
    # x, y = placeholders
    # loss, accuracy = metrics
    with tf.Session() as sess:
        sess.run(init)
        for i in range(train_steps):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            loss_val, acc_val, _ = sess.run(
                [loss, accuracy, train_op],
                feed_dict={x: batch_data,
                           y: batch_labels})
            if (i + 1) % 500 == 0:
                print('[Train] Step: {}, loss: {}, acc: {}'.format(i + 1, loss_val, acc_val))
            if (i + 1) % 5000 == 0:
                test_data = CiferData(test_filenames, False)
                all_test_acc_val = []
                for j in range(test_steps):
                    test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                    test_acc_val = sess.run(
                        [accuracy], feed_dict={x: test_batch_data,
                                               y: test_batch_labels
                                               })
                    all_test_acc_val.append(test_acc_val)
                test_acc = np.mean(all_test_acc_val)
                print('[Test] Step: {}, acc: {}'.format(i + 1, test_acc))
