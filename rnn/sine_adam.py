#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-4
# @Author  : wyxiang
# @File    : sine_adam.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2
#
# 使用CUDA训练

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
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

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, future=0):
        """
        前向传播
        :param input: 输入, size = (batch, 999, input_size)
        :param future: 在输入的基础上向前预测的步数
        :return: 预测结果
        """

        outputs = []
        # 初始化参数
        h_t = torch.zeros(1, input.size(0), 51, dtype=torch.double).cuda()
        c_t = torch.zeros(1, input.size(0), 51, dtype=torch.double).cuda()
        # 对于输入中的点，使用真实值作为输入
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            out, (h_t, c_t) = self.rnn(input_t, (h_t, c_t))
            output = self.linear(out)
            outputs += [output]
        # 对于预测值，使用上一个time_step预测的点作为输入
        for i in range(future):
            out, (h_t, c_t) = self.rnn(output, (h_t, c_t))
            output = self.linear(out)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class SineDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data):
        self.input = torch.from_numpy(data[:, :-1])
        self.target = torch.from_numpy(data[:, 1:])

    def __getitem__(self, index):
        return self.input[index].view(-1, 1), self.target[index].view(-1, 1)

    def __len__(self):
        return self.input.size(0)


def generate_data():
    """
    生成训练/测试数据
    :return: train_dataset, test_dataset
    """
    # 复制自generate_wave_sine.py
    np.random.seed(2)
    T, L, N = 20, 1000, 100
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    return SineDataset(data[3:, :]), SineDataset(data[:3, :])


if __name__ == '__main__':
    writer = SummaryWriter()

    # 生成训练/测试数据集
    train_dataset, test_dataset = generate_data()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=97,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=3)

    # 建立模型
    model = RNN(1, 51, 1, 1).cuda()
    model.double()
    loss_func = nn.MSELoss().cuda()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    for i in range(1000):
        for idx, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            out = model(input)
            loss = loss_func(out, target)
            writer.add_scalar('train/loss', loss.item(), i + 1, walltime=i + 1)
            print('epoch {}, loss= {}'.format(i + 1, loss.item()))
            loss.backward()
            optimizer.step()
        # 测试模型，并向前预测1000个点
        with torch.no_grad():
            for idx, (test_input, test_target) in enumerate(test_loader):
                test_input, test_target = test_input.cuda(), test_target.cuda()
                future = 1000
                pred = model(test_input, future=future)
                loss = loss_func(pred[:, :-future], test_target)
                writer.add_scalar('test/loss', loss.item(), i + 1, walltime=i + 1)
                print('epoch {}, test loss= {}'.format(i + 1, loss.item()))
                y = pred.detach().cpu().numpy()
        writer.close()  # 立刻刷新
        if (i + 1) % 50 != 0:
            continue
        # 绘制结果图
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.show()
