import os
import tensorflow as tf
import numpy as np
from PIL import Image
from cnn.style_transfer.vgg16net import VGGNet

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


def gram_matrix(x):
    """
    Calculates gram matrix.
    :param x: features extracted from VGG Net. shape: [1, width, height, ch]
    :return:
    """
    b, w, h, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h * w, ch])
    # [h*w, ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch]
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
    return gram


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
style_gram = [gram_matrix(feature) for feature in style_features]

result_style_features = [
    # vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]
result_style_gram = [gram_matrix(feature) for feature in result_style_features]

# 计算内容损失
content_loss = tf.zeros(1, tf.float32)
# shape: [1, width, height, channel]
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

# 计算风格损失
style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])

loss = content_loss * lambda_c + style_loss * lambda_s
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_value, content_loss_value, style_loss_value, _ \
            = sess.run([loss, content_loss, style_loss, train_op],
                       feed_dict={content: content_val, style: style_val})
        print('step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f' %
              (step+1, loss_value[0], content_loss_value[0], style_loss_value[0]))
        result_img_path = os.path.join(output_dir, 'result-%05d.jpg' % (step+1))
        result_val = result.eval(sess)[0]
        result_val = np.clip(result_val, 0, 255)
        img_arr = np.asarray(result_val, np.uint8)
        img = Image.fromarray(img_arr)
        img.save(result_img_path)
