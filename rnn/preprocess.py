#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-4
# @Author  : wyxiang
# @File    : preprocess.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

import numpy as np
import codecs

def get_dict(file):
    word2vec = {}
    word2vec_file = codecs.open(file, 'r', 'utf-8')
    word2vec_text = word2vec_file.readlines()
    for line in word2vec_text:
        s = line.strip().split()
        if len(s) == 0: continue
        vec = [float(v) for v in s[1:]]
        word2vec[s[0].lower()] = vec
    return word2vec


def get_vec(file, label, word2vec):
    file = codecs.open(file, 'r', 'Windows-1252')
    text = file.readlines()
    zero_padding = [0.0] * 50
    max_len, data = 0, []
    for line in text:
        words = line.lower().strip().split()
        if len(words) == 0: continue
        max_len = len(words) if len(words) > max_len else max_len
        words_vec = []  # 行向量
        for word in words:
            if word == '': continue
            if word not in word2vec:
                words_vec.append(zero_padding)
                continue
            words_vec.append((word2vec[word]))
        data.append((words_vec, label))
    return data, max_len


file_pos = 'text/rt-polarity.pos'
file_neg = 'text/rt-polarity.neg'
file_dict = 'text/glove.6B.50d.txt'

word2vec = get_dict(file_dict)
print('word2vec finish')

pos_data, max_len1 = get_vec(file_pos, 1, word2vec)
print('pos finish')

neg_data, max_len2 = get_vec(file_neg, 0, word2vec)
print('neg finish')

max_len = max_len1 if max_len1 > max_len2 else max_len2
zero_padding = [0.0] * 50

# 补齐数据
data = []
data.extend(pos_data)
data.extend(neg_data)
for d, l in data:
    d.extend([zero_padding] * (max_len - len(d)))
print('padding finish')

# 随机打乱数据
np.random.shuffle(data)

# 划分训练集和测试集
train_len = int(len(data) * 0.8)
train_data = data[:train_len]
test_data = data[train_len:]

np.save('text/glove_train.npy', train_data)
np.save('text/glove_test.npy', test_data)
