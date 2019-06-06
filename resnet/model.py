#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-26
# @Author  : wyxiang
# @File    : model.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

from torch import nn
from torch.nn import functional as F
import torch


class ResidualBlock(nn.Module):
    """
    ResidualBlock实现
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None, device=torch.device('cpu')):
        super(ResidualBlock, self).__init__()
        self.device = device
        # 主干部分为为conv-bn-relu-conv-bn
        self.blocks = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # 恒等映射部分
        self.shortcut = shortcut

    def forward(self, x):
        out = self.blocks(x)
        # 恒等映射计算
        identity = x if self.shortcut is None else self.shortcut(x)
        # 计算channel差值，并用0 pad
        diff = out.size()[1] - identity.size()[1]
        if diff > 0:
            # padding
            b, c, h, w = out.size()
            identity = torch.cat((identity, torch.zeros(b, diff, h, w).to(self.device)), 1)
        out += identity
        return F.relu(out)


class ResNet(nn.Module):
    """
    实现RenNet18模块
    ResNet18包含4个layer,每个layer包含2个residual block
    用子module实现residual block , 用 _make_layer 函数实现layer
    """
    def __init__(self, num_classes=10, device=torch.device('cpu')):
        super(ResNet, self).__init__()
        self.device = device
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer,每个layer各两个block，除layer1外，第一个block时均会有downsample
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

        # 初始化参数，参考自pytorch resnet model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含2个residual block
        # 恒等映射channel不匹配时的pooling层
        shortcut = nn.MaxPool2d(kernel_size=1, stride=stride)
        # 或者考虑1x1 conv?
        # shortcut =  nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=stride)
        # # 或者考虑conv + bn
        # shortcut = nn.Sequential(
        #     nn.Conv2d(inchannel, outchannel, 1, stride),
        #     nn.BatchNorm2d(outchannel),
        # )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut, device=self.device))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, device=self.device))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
