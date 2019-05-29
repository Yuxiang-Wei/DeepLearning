#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-23
# @Author  : wyxiang
# @File    : dataset.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

from PIL import Image
import os
import numpy as np
import pickle

import torch.utils.data as data


class CIFAR10(data.Dataset):
    """
    读取CIFAR10数据集
    """
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5'
    ]

    test_list = [
        'test_batch',
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        self.imgs = []
        self.labels = []

        # load the picked numpy arrays
        for file_name in data_list:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                print('Missing file {} in {}'.format(file_name, file_path))
                exit()
            with open(file_path, 'rb') as f:
                dict = pickle.load(f, encoding='latin1')
                self.imgs.append(dict['data'])
                if 'labels' in dict:
                    self.labels.extend(dict['labels'])
                else:
                    self.labels.extend(dict['fine_labels'])

        # reshape the image to 32x32
        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs)
