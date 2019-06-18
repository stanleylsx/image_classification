import time
import numpy as np
import tensorflow as tf


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
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
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
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        """
        Builds fully-connected layer.
        :param activation:
        :param x:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            else:
                return activation(h)

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

    def build(self, x_rgb):
        """
        Build VGG16 network structure.
        :param x_rgb: [1, 224, 224, 3]
        :return:
        """
        start_time = time.time()
        print('building model...')
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(x_bgr, b'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, b'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, b'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, b'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, b'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, b'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, b'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, b'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, b'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, b'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, b'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, b'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, b'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, b'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, b'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, b'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, b'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, b'pool5')

        '''
        self.flatten5 = self.flatten_layer(self.pool5, b'flatten')
        self.fc6 = self.fc_layer(self.flatten5, b'fc6')
        self.fc7 = self.fc_layer(self.fc6, b'fc7')
        self.fc8 = self.fc_layer(self.fc7, b'fc8', activation=None)
        self.prob = tf.nn.softmax(self.fc8, name=b'prob')
        '''

        print('building model finished: %4ds' % (time.time()-start_time))


if __name__ == '__main__':
    vgg16_npy_path = '../../data/style_transfer_data/vgg16.npy'
    data_dict = np.load(vgg16_npy_path, encoding='bytes', allow_pickle=True).item()
    vgg16_for_result = VGGNet(data_dict)
    content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
    vgg16_for_result.build(content)
