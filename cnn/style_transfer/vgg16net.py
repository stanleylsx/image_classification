import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class VGGNet:
    """
    Builds VGG-16 net structure, load parameters from pre-train models.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        """
        Builds convolutions layer.
        :param x:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='same')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h

    def pooling_layer(self, x, name):
        """
        Builds pooling layer.
        :param x:
        :param name:
        :return:
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='same', name=name)

    def fc_layer(self, x, name):
        """
        Builds fully-connected layer.
        :param x:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            h = tf.nn.relu(h)
            return h

    def flatten_layer(self, x, name):
        """
        Builds flatten layer.
        :param x:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            return x


if __name__ == '__main__':
    vgg16_data = np.load('../../data/vgg16.npy', encoding='bytes', allow_pickle=True)
    vgg16_dict = vgg16_data.item()
    conv1_1 = vgg16_dict[b'conv1_1']
    print(len(conv1_1))
