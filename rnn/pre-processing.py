# 分词
# 词语 -> id
# matrix -> [|v|, embed_size]
# 词语A -> id

import sys
import os
import jieba

# input files
train_file = '../data/text_classification_data/cnews.train.txt'
val_file = '../data/text_classification_data/cnews.val.txt'
test_file = '../data/text_classification_data/cnews.test.txt'

# output files
seg_train_file = './cnews.train.seg.txt'
seg_val_file = './cnews.val.seg.txt'
seg_test_file = './cnews.test.seg.txt'

vocal_file = './cnews.vocab.txt'
category_file = './cnews.category.txt'


def generate_seg_file(input_file, output_seg_file):
    """
    Segment the sentences in each line in input_file
    :param input_file:
    :param output_seg_file:
    :return:
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line)


generate_seg_file(train_file, seg_train_file)
generate_seg_file(val_file, seg_val_file)
generate_seg_file(test_file, seg_test_file)


def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, conntent = line.strip('\r\n').split('\t')
        for word in conntent.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # [(word, frequency),...,()]
    sorted_word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    with open(output_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<UNK>\t1000000')
