#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-06-08
# @Author  : wyxiang
# @File    : model.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

import torch.nn as nn


class Generator(nn.Module):
    """
    生成器模型
    nz：  噪声大小
    ngf： 中间节点大小
    isize： 输出大小
    """

    def __init__(self, nz, ngf, isize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, isize),
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    """
    判别器模型
    isize： 输入大小
    ndf： 中间节点大小
    wgan: 是否为wgan模型，其决定最后一层是否使用sigmoid层
    """

    def __init__(self, isize, ndf, wgan=False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.wgan = wgan
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        if not self.wgan:
            output = self.sigmoid(output)
        return output
