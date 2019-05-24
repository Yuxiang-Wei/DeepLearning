#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-23
# @Author  : wyxiang
# @File    : resnet.py
# @Env: Ubuntu16.04 Python3.6
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from optparse import OptionParser
from dataloader import CIFAR10

# 判断CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

model_dir = './checkpoints/lastest.pkl'  # 模型存放地址


class AlexNet(nn.Module):
    """
    AlexNet,参考自torchvision.models.alexnet
    修改如下：
    1. 将前两个Conv层的kernel size降为3，padding设为1，保证图片尺寸不变
    2. 将所有的pooling层kernel_size降为2，防止图片过小
    3. 移除average pooling层
    4. 降低最后全连接层节点个数，加速训练
    5. 其余均保持不变
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32 -> 16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16 -> 8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # 展开为一维
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x


def load_data(train=True, batch_size=50, shuffle=False):
    """
    加载数据集
    :param train: 训练集 or 测试集
    :param batch_size: batch的大小
    :return: 返回加载好的Dataloader
    """
    if train:
        # 针对训练数据随机打乱
        shuffle = True

    # 加载CIFAR10数据集，若不存在则下载
    dataset = CIFAR10('./cifar-10-batches-py',
                      train=train,
                      transform=transforms.ToTensor())
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader


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

    # 加载训练集，测试集
    train_loader = load_data(train=True, batch_size=batch_size)
    test_loader = load_data(train=False, batch_size=batch_size)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    for i in range(1, epochs + 1):
        # model设置为训练模式
        model.train()
        # 遍历每个batch
        train_loss = 0.0
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
        test_loss, test_accuracy = validate(model, test_loader, criterion)

        # 保存模型参数, 这里设定每个epoch保存一次
        save_dir = './checkpoints'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, 'lastest.pkl'))


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
    return val_loss, accuracy


def get_args():
    """
    解析命令行参数
    :return: 参数列表
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=50,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-m', '--load', dest='load', default=None, help='load model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    model = AlexNet().to(device)
    print(model)
    if args.load is not None:
        if not os.path.exists(model_dir):
            print('model not found')
        else:
            print('load exist model')
            model.load_state_dict(torch.load(model_dir))
    train(model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)
