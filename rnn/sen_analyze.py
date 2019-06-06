#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-4
# @Author  : wyxiang
# @File    : sen_analyze.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


class RNN(nn.Module):
    """
    RNN模型
    由一个单层LSTM网络 + fc层构成
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        # 取最后一个time_step的输出作为输出
        out = self.out(r_out[:, -1, :])
        return out


class TextDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input, label = self.data[index]
        return torch.FloatTensor(input), label

    def __len__(self):
        return len(self.data)


def get_data():
    """
    从文件读取数据集
    :return: train_dataset, test_dataset
    """
    train_data = np.load('text/glove_train.npy')
    test_data = np.load('text/glove_test.npy')
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    return train_dataset, test_dataset


def train(epochs=1000, lr=0.005, device=torch.device('cuda'), log_interval=10):
    """
    训练模型
    :param epochs: 迭代轮数
    :param lr: 学习率
    :param device: GPU or CPU
    :param log_interval: 打印间隔
    :return: None
    """
    # 0.005, 0.01, 0.1 0.00001
    writer = SummaryWriter()

    # 构建模型
    model = RNN(50, 64, 2, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 使用交叉熵损失函数
    loss_func = nn.CrossEntropyLoss().to(device)
    # 获取数据集
    train_dataset, test_dataset = get_data()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=200,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=20)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, label)
            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
        print('Train Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader)))
        val_loss, val_acc = validate(model, test_loader, loss_func, device)
        writer.add_scalar('train/loss', train_loss, epoch + 1, walltime=epoch + 1)
        writer.add_scalar('test/loss', val_loss, epoch + 1, walltime=epoch + 1)
        writer.add_scalar('test/accuracy', val_acc, epoch + 1, walltime=epoch + 1)
        writer.close()


def validate(model, data_loader, loss_func, device):
    """
    测试模型在给定数据上的正确率
    :param model: 模型
    :param data_loader: 给定数据集
    :param loss_func: 损失函数
    :return: loss, accuracy
    """
    # 设定模型为执行模式
    model.eval()
    val_loss, correct = 0, 0
    for data, label in data_loader:
        # 若GPU可用，拷贝数据至GPU
        data, label = data.to(device), label.to(device)
        # 前向传播
        output = model(data)
        # 计算loss
        val_loss += loss_func(output, label).item()
        # 获得概率最大的下标，即分类结果
        pred = output.data.max(1)[1]
        # 计算正确个数
        correct += pred.eq(label.data).sum()
    val_loss /= len(data_loader)
    accuracy = 100. * correct.to(torch.float32) / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(data_loader.dataset), accuracy))
    return val_loss, accuracy


if __name__ == '__main__':
    train()
