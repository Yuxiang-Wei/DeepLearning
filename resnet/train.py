#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-26
# @Author  : wyxiang
# @File    : train.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from dataset import CIFAR10
from model import ResNet
import argparse

model_dir = './checkpoints/lastest.pkl'  # 模型存放地址


def load_data(isTrain, batch_size=64, shuffle=False):
    dataset = CIFAR10('./cifar-10-batches-py',
                      train=isTrain,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)

    return data_loader

def train(model, epochs=10, batch_size=64, lr=1e-2, log_interval=20, device=torch.device('cpu')):
    """
        训练网络模型
        :param model: 待训练网络
        :param epochs: 迭代次数
        :param batch_size: batch size
        :param lr: 学习率
        :param log_interval: 打印loss的间隔次数
        :return:
        """

    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    # 加载训练集，测试集
    train_loader = load_data(isTrain=True, batch_size=batch_size, shuffle=True)
    test_loader = load_data(isTrain=False, batch_size=batch_size)
    # 使用Adam优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(1, epochs + 1):
        # model设置为训练模式
        model.train()
        # 遍历每个batch
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            # 将梯度缓存置0
            optimizer.zero_grad()
            # 执行一次前向传播
            output = model(data)
            # 计算loss
            loss = criterion(output, target)
            train_loss += loss
            # 反向传播
            loss.backward()
            # 更新权值
            optimizer.step()
            # 打印loss信息
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))

        # Test the model
        test_loss, test_accuracy = validate(model, test_loader, criterion, device)


def validate(model, data_loader, criterion, device):
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
    return val_loss, accuracy


def boolean(s):
    return s.lower() in ('true', 't', 'yes', 'y', '1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=boolean, default=False,
                        help='use gpu or cpu ? y is gpu and n is cpu')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int,
                      help='number of epochs')
    parser.add_argument('--batch_size', dest='batchsize', default=256,
                      type=int, help='batch size')
    parser.add_argument('--lr', dest='lr', default=1e-2,
                      type=float, help='learning rate')
    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ResNet(device=device).to(device)

    train(model, lr=args.lr, epochs=args.epochs, batch_size=args.batchsize, device=device)
