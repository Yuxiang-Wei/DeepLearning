#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-16
# @Author  : wyxiang
# @File    : model.py
# @Env: Ubuntu16.04 Python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

# 判断CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 64 * 6 * 6)
        x = self.classifier(x)
        return x


def load_data(train=True, batch_size=50):
    """
    加载数据集
    :param train: 训练集 or 测试集
    :param batch_size: batch的大小
    :return: 返回加载好的Dataloader
    """

    # 加载MNIST数据集，若不存在则下载
    dataset = datasets.CIFAR10('./data',
                               train=train,
                               download=True,
                               transform=transforms.ToTensor())
    if train:
        data_size = len(dataset)
        # 分为训练集和验证集
        train, val = torch.utils.data.random_split(dataset, [int(data_size * 0.7), int(data_size * 0.3)])
        # 随机打乱训练集
        train_loader = DataLoader(dataset=train,
                                  batch_size=batch_size,
                                  shuffle=True)
        # 准备验证集
        val_loader = DataLoader(dataset=val,
                                batch_size=batch_size,
                                shuffle=True)
        return train_loader, val_loader
    else:
        # 准备测试集
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        return test_loader, None

def train(model, epochs, batch_size, lr, log_interval=200):
    """
    训练网络模型
    :param model: 待训练网络
    :param epochs: 迭代次数
    :param batch_size: batch size
    :param lr: 学习率
    :param log_interval: 打印loss的间隔次数
    :return:
    """

    # 加载训练集，验证集
    train_loader, val_loader = load_data(train=True, batch_size=batch_size)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    for i in range(1, epochs + 1):
        # model设置为训练模式
        model.train()
        # 遍历每个batch
        for batch_idx, (data, target) in enumerate(train_loader):
            # 若GPU可用，拷贝数据至GPU
            data = data.to(device)
            target = target.to(device)
            # 将梯度缓存置0
            optimizer.zero_grad()
            # 执行一次前向传播
            output = model(data)
            # 计算loss
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            # 更新权值
            optimizer.step()
            # 打印loss信息
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))
        validate(model, val_loader, criterion)

def validate(model, data_loader, criterion):
    """
    测试模型在给定数据上的正确率
    :param model: 模型
    :param data_loader: 给定数据集
    :param criterion: 损失函数
    :return:
    """
    # 设定模型为执行模式
    model.eval()
    val_loss, correct = 0, 0
    for data, target in data_loader:
        # 若GPU可用，拷贝数据至GPU
        data = data.to(device)
        target = target.to(device)
        # 前向传播
        output = model(data)
        # 计算loss
        val_loss += criterion(output, target).data.item()
        # 获得概率最大的下标，即分类结果
        pred = output.data.max(1)[1]
        # 计算正确个数
        correct += pred.eq(target.data).cpu().sum()
    val_loss /= len(data_loader)
    accuracy = 100. * correct.to(torch.float32) / len(data_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(data_loader.dataset), accuracy))
if __name__ == '__main__':
    model = AlexNet().to(device)
    print(model)
    train(model, epochs=20, batch_size=50, lr=1e-3)