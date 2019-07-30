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


def residual_block(input, output_channel):
    """
    residual connection implementation
    :param input:
    :param output_channel:
    :return:
    """
    input_channel = input.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2, 2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1, 1)
    else:
        raise Exception('input channel can not match output channel')
    conv1 = tf.layers.conv2d(input,
                             output_channel,
                             (3, 3),
                             strides=strides,
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv1')

    conv2 = tf.layers.conv2d(conv1,
                             output_channel,
                             (3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv2')

    if increase_dim:
        # [None, image_width, image_height, channel] -> [,,,channel*2]
        pooled_x = tf.layers.average_pooling2d(input,
                                               (2, 2),
                                               (2, 2),
                                               padding='valid')

        # 池化操作不会增加通道数，故需要手动的增加通道数
        padded_x = tf.pad(pooled_x,
                          [
                              [0, 0],
                              [0, 0],
                              [0, 0],
                              [input_channel // 2, input_channel // 2]
                          ])
    else:
        padded_x = input
    output_x = conv2 + padded_x
    return output_x


def res_net(x_input, num_residual_blocks, num_filter_base, class_num):
    """
    residual network implementation
    :param x_input:
    :param num_residual_blocks: eg: [3, 4, 6, 3]
    :param num_filter_base:
    :param class_num:
    :return:
    """
    num_subsampling = len(num_residual_blocks)
    layers = []
    # x [None, width, height, channel] -> [width, height, channel]
    input_size = x_input.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x_input,
                                 num_filter_base,
                                 (3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=tf.nn.relu,
                                 bias_initializer=tf.zeros_initializer(),
                                 name='conv0')
        layers.append(conv0)
    # num_subsampling = 4 , sample_id = [0, 1, 2, 3]
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope('conv%d_%d' % (sample_id, i)):
                conv = residual_block(layers[-1], num_filter_base * (2 ** sample_id))
                layers.append(conv)

    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] == [input_size[0] / multiplier, input_size[1] / multiplier,
                                                    num_filter_base * multiplier]

    # layer[-1].shape : [None, width, height, channel]
    # kernal_size : image_width, image_height
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    logits = tf.layers.dense(global_pool, class_num)
    layers.append(logits)
    return layers[-1]


x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])
# [None] eg: [0,5,6,3]
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32*32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

y_ = res_net(x_image, [2, 3, 2], 32, 10)

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

    with tf.Session() as sess:
        sess.run(init)
        for i in range(train_steps):
            batch_data, batch_labels = train_data.next_batch(batch_size)
            loss_val, acc_val, _ = sess.run([loss, accuracy, train_op], feed_dict={x: batch_data, y: batch_labels})
            if (i + 1) % 500 == 0:
                print('[Train] Step: {}, loss: {}, acc: {}'.format(i + 1, loss_val, acc_val))
            if (i + 1) % 5000 == 0:
                test_data = CiferData(test_filenames, False)
                all_test_acc_val = []
                for j in range(test_steps):
                    test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                    test_acc_val = sess.run([accuracy], feed_dict={x: test_batch_data, y: test_batch_labels})
                    all_test_acc_val.append(test_acc_val)
                test_acc = np.mean(all_test_acc_val)
                print('[Test] Step: {}, acc: {}'.format(i + 1, test_acc))
