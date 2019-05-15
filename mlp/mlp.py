#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-5-10
# @Author  : wyxiang
# @File    : mlp.py
# @Env: Ubuntu16.04 Python3.6 Pytorch0.4.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from optparse import OptionParser
import os

# 判断CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

img_size = 28
output_size = 10
model_dir = './checkpoints/lastest.pkl' # 模型存放地址


# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(img_size * img_size, 512)  # 隐层1, [784,1] -> [512,1]
        self.linear2 = nn.Linear(512, 256)  # 隐层2, [512,1] -> [256,1]
        self.linear3 = nn.Linear(256, output_size)  # 隐层3 [256,1] -> [10,1]

    def forward(self, x):
        # 将图片展开成一维，已匹配隐层1的输入大小，即[batch, 1, img_size, img_size] -> [batch, img_size*img_size]
        x = x.view(-1, img_size * img_size)
        x = F.relu(self.linear1(x))  # 经过隐层1并使用relu函数激活
        x = F.relu(self.linear2(x))  # 经过隐层2并使用relu函数激活
        return F.log_softmax(self.linear3(x), dim=1)  # 经过隐层3，并最终经过一个log_softmax层做分类输出


def load_data(train=True, batch_size=50):
    """
    加载数据集
    :param train: 训练集 or 测试集
    :param batch_size: batch的大小
    :return: 返回加载好的Dataloader
    """

    # 加载MNIST数据集，若不存在则下载
    dataset = datasets.MNIST('./data',
                             train=train,
                             download=True,
                             transform=transforms.ToTensor())
    if train:
        # 分为训练集和验证集
        train, val = torch.utils.data.random_split(dataset, [50000, 10000])
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
        # 保存模型参数, 这里设定每个epoch保存一次
        save_dir = './checkpoints'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, 'lastest.pkl'))
        # 测试模型在验证集上的正确率
        validate(model, val_loader, criterion)


def test(model, batch_size):
    """
    测试模型在测试集上的正确率
    :param model: 模型
    :param batch_size: batch size
    :return:
    """
    test_loader, _ = load_data(train=False, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    validate(model, test_loader, criterion)


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


def get_args():
    """
    解析命令行参数
    :return: 参数列表
    """
    parser = OptionParser()
    parser.add_option('-t', '--train', action="store_true", dest='train', default=True,
                      help='train model')
    parser.add_option("-v", '--test', action="store_false", dest="train",
                      help='test model')
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=50,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='lr', default=0.001,
                      type='float', help='learning rate')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    model = MLP().to(device)
    if args.train:
        print(model)
        train(model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)
    else:
        if not os.path.exists(model_dir):
            print('model not found')
        else:
            model.load_state_dict(torch.load(model_dir))
            test(model, batch_size=args.batchsize)
