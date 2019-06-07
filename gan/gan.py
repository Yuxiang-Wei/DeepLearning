#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-6
# @Author  : wyxiang
# @File    : gan.py
# @Env: Ubuntu16.04 Python3.6

import torch.nn as nn
import torch
from torch import optim
import scipy.io as sciio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(10, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output

class PointDataset(Dataset):
    def __init__(self, data):
        self.input = data
    def __getitem__(self, index):

        return torch.from_numpy(self.input[index])

    def __len__(self):
        return len(self.input)

def get_data():
    mat = sciio.loadmat('points.mat')
    data = np.vstack((mat['xx'])).astype('float32')
    train_dataset = PointDataset(data)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    return data, train_loader


netG = Generator()
netD = Discriminator()

criterion = nn.BCELoss()
real_label = 1
fake_label = 0
lr = 0.00018
epochs = 200
batch_size = 400


fixed_noise = torch.randn(batch_size, 10)

# setup optimizer
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

true_data, data_loader = get_data()

for epoch in range(epochs):
    outputs = []
    for i, data in enumerate(data_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label)

        output1 = netD(real_cpu)
        errD_real = criterion(output1, label)
        D_x = output1.mean().item()

        # train with fake
        # noise = fixed_noise
        noise = torch.randn(batch_size, 10)
        fake = netG(noise)
        outputs.append(fake.data)
        label_fake = torch.zeros((batch_size,))
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label_real = torch.ones((batch_size,))
        output = netD(fake)
        errG = criterion(output, label_real)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(data_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    plt.scatter(true_data[:, 0], true_data[:, 1], c='b')
    for fake in outputs:
        x = fake.numpy()
        plt.scatter(x[:, 0], x[:, 1], c='r')
    plt.show()