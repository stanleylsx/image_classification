import os
import tensorflow as tf
import numpy as np
from PIL import Image
from .vgg16net import VGGNet

vgg16_npy_path = '../../data/style_transfer_data/vgg16.npy'
content_img_path = '../../data/style_transfer_data/palace.jpg'
style_img_path = '../../data/style_transfer_data/starry_sky.jpeg'

num_steps = 100
learning_rate = 10
lambda_c = 0.1
lambda_s = 500

output_dir = './run_style_transfer'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def initial_result(shape, mean, stddev):
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def read_image(img_name):
    img = Image.open(img_name)
    np_img = np.array(img)  # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32)  # (1, 224, 224, 3)
    return np_img


result = initial_result((1, 224, 224, 3), 127.5, 20)
content_val = read_image(content_img_path)
style_val = read_image(style_img_path)

content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

data_dict = np.load(vgg16_npy_path, encoding='bytes', allow_pickle=True).item()
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)

# 内容特征越低层越好
content_features = [
    vgg_for_content.conv1_2,
    # vgg_for_content.conv2_2,
    # vgg_for_content.conv3_3,
    # vgg_for_content.conv4_3,
    # vgg_for_content.conv5_3
]

result_content_features = [
    vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    # vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

# 风格特征越高层越好
# feature_size, [1, width, height, channel]
style_features = [
    # vgg_for_style.conv1_2,
    # vgg_for_style.conv2_2,
    # vgg_for_style.conv3_3,
    vgg_for_style.conv4_3,
    # vgg_for_style.conv5_3
]

result_style_features = [
    # vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

# 计算内容损失
content_loss = tf.zeros(1, tf.float32)
# shape: [1, width, height, channel]
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

# 计算风格损失



