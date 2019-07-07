# TODO.1 词表封装:
#   api: sentence2id(text_sentence): 句子转换id
# TODO.2 类别封装:
#   api: category2id(text_category).
# TODO.3 数据集封装代码
#   api: next_batch(batch_size)
# TODO.4 训练流程代码
# TODO.5 构建计算图-LSTM模型
#   1.1 embedding
#   1.2 LSTM
#   1.3 fc
#   1.4 train_op

import tensorflow as tf
import os
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.INFO)


def get_default_params():
    """
    lstm的参数设置
    :return:
    """
    return tf.contrib.training.HParams(
        num_embedding_size=16,  # embedding编码的长度
        num_timesteps=50,
        num_lstm_nodes=[32, 32],  # 每层lstm的神经单元个数
        num_lstm_layers=2,  # lstm的层次
        num_fc_nodes=32,  # 全连层神经单元数目
        batch_size=100,
        clip_lstm_grads=1.0,  # 控制lstm梯度大小，防止梯度爆炸
        learning_rate=0.001,  # 学习率
        num_word_threshold=10,  # 出现词频10次以下的词语不加入模型训练中
    )


class Vocab:
    """
    词表封装类
    """
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return word_ids


class CategoryDict:
    """
    类别的封装
    """
    def __init__(self, filename):
        self._category_to_id = {}
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def category_to_id(self, category):
        if category not in self._category_to_id:
            raise Exception('%s is not in our category list'.format(category))
        return self._category_to_id[category]

    def size(self):
        return len(self._category_to_id)


class TextDataSet:
    """
    文本数据的封装
    """
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        # matrix
        self._inputs = []
        # vector
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s' % filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            id_label = self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0: self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception('batch_size: %d is too long' % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs


def create_model(hps, vocab_size, num_categories):
    # 定义计算图
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    # dropout prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    # 定义embedding层
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    # 定义lstm层
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hps.num_lstm_nodes[i],  # 大小
                state_is_tuple=True
            )
            # 加入dropout
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=keep_prob
            )
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state = cell.zero_state(batch_size, tf.float32)
        # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state=initial_state)
        last = rnn_outputs[:, -1, :]

    # 定义全连接层
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last, hps.num_fc_nodes, activation=tf.nn.relu, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout, num_categories, name='fc2')

    # 计算损失函数
    with tf.variable_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # train_op定义
    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % var.name)
        # 截断梯度
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads
        )
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return (inputs, outputs, keep_prob), (loss, accuracy), (train_op, global_step)


if __name__ == '__main__':
    hps = get_default_params()
    train_file = './cnews.train.seg.txt'
    val_file = './cnews.val.seg.txt'
    test_file = './cnews.test.seg.txt'
    vocab_file = './cnews.vocab.txt'
    category_file = './cnews.category.txt'
    output_folder = './run_text_rnn'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    vocab = Vocab(vocab_file, hps.num_word_threshold)
    vocab_size = vocab.size()
    tf.logging.info('vocab_size: %d' % vocab_size)

    category_vocab = CategoryDict(category_file)
    num_categories = category_vocab.size()
    tf.logging.info('num_categories: %d' % num_categories)

    train_dataset = TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps)
    val_dataset = TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
    test_dataset = TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)

    placeholders, metrics, others = create_model(hps, vocab_size, num_categories)
    inputs, outputs, keep_prob = placeholders
    loss, accuracy = metrics
    train_op, global_step = others

    init_op = tf.global_variables_initializer()
    train_keep_prob_value = 0.8
    test_keep_prob_value = 1.0
    num_train_steps = 10000

    # Train: 99.7%
    # Valid: 92.7%
    # Test: 93.25%
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(num_train_steps):
            batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)
            outputs_val = sess.run([loss, accuracy, train_op, global_step], feed_dict={
                inputs: batch_inputs,
                outputs: batch_labels,
                keep_prob: train_keep_prob_value
            })
            loss_val, accuracy_val, _, global_step_val = outputs_val
            if global_step_val % 100 == 0:
                tf.logging.info('Step: %5d, loss: %3.3f, accuracy: %3.5f' % (global_step_val, loss_val, accuracy_val))









