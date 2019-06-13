#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-06-08
# @Author  : wyxiang
# @File    : main.py
# @Env: Ubuntu16.04 Python3.7 pytorch1.0.1.post2

import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import scipy.io as sciio
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from model import Generator, Discriminator
from tensorboardX import SummaryWriter


class PointDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data):
        self.input = data

    def __getitem__(self, index):
        return torch.from_numpy(self.input[index])

    def __len__(self):
        return len(self.input)


def get_data(data_root, batch_size):
    """
    读取数据集
    :param data_root: 数据根目录
    :param batch_size: batch 大小
    :return: data, train_loader
    """
    data_path = os.path.join(data_root, 'points.mat')
    if not os.path.exists(data_path):
        print('dataset not exist')
        exit()
    mat = sciio.loadmat(data_path)
    data = np.vstack((mat['xx'])).astype('float32')
    train_dataset = PointDataset(data)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    return data, train_loader


def calc_gradient_penalty(netD, real_data, fake_data):
    """
    计算梯度惩罚函数，用于WGAN-GP， 借鉴自https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    :param netD: 判别器网络
    :param real_data: 真实数据
    :param fake_data: 生成数据
    :return: gradient_penalty
    """
    # 生成混合系数
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def visualize(netG, netD, real_data, epoch, save_path, mode='gan', cuda=True):
    """
    可视化训练结果
    :param netG: 生成器
    :param netD: 判别器
    :param real_data: 真实数据
    :param epoch: 第几个epoch
    :param save_path: 图片存储路径
    :param mode: 当前模型类型 gan/wgan/wgan-gp
    :param cuda: 是否使用cuda
    :return: none
    """
    netG.eval()
    netD.eval()
    # 生成噪声
    noise = torch.randn(1000, opt.nz)
    if cuda:
        noise = noise.cuda()
    # 生成数据
    fake = netG(noise)
    fake_np = fake.cpu().detach().numpy()

    # 获取原始数据和生成数据的最大最小范围，方便画图
    img_xl, img_xh = min(np.min(fake_np[:, 0]), -0.5), max(np.max(fake_np[:, 0]), 1.5)
    img_yl, img_yh = min(np.min(fake_np[:, 1]), 0), max(np.max(fake_np[:, 1]), 1)
    # 对图片区域进行均匀采样
    a_x = np.linspace(img_xl, img_xh, 200)
    a_y = np.linspace(img_yl, img_yh, 200)
    u = [[x, y] for y in a_y[::-1] for x in a_x[::-1]]
    u = np.array(u)
    if cuda:
        u1 = torch.FloatTensor(u).cuda()
    # 将采样值送入判别器，得结果
    outs = netD(u1)
    outs_np = outs.cpu().detach().numpy()
    # 绘制判别器结果，为黑白热度图，并存储
    plt.cla()
    plt.clf()
    d_path = os.path.join(save_path, 'discriminator')
    if not os.path.exists(d_path):
        os.makedirs(d_path)
    plt.imshow(outs_np.reshape(200, 200), extent=[img_xl, img_xh, img_yl, img_yh], cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(d_path, 'epoch{}.png'.format(epoch)))

    # 绘制生成器结果，并存储
    plt.cla()
    plt.clf()
    if mode == 'gan':
        c = ['w' if x >= 0.4999 else 'black' for x in outs_np]
        plt.scatter(u[:, 0], u[:, 1],
                    c=c, alpha=0.3, marker='s')
    else:
        plt.imshow(outs_np.reshape(200, 200), extent=[img_xl, img_xh, img_yl, img_yh], cmap='gray')
        plt.colorbar()
    plt.scatter(real_data[:, 0], real_data[:, 1], c='b')
    plt.scatter(fake_np[:, 0], fake_np[:, 1], c='r')
    g_path = os.path.join(save_path, 'generator')
    if not os.path.exists(g_path):
        os.makedirs(g_path)
    plt.savefig(os.path.join(g_path, 'epoch{}.png'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./', help='path to dataset')
    parser.add_argument('--batchSize', type=int, default=400, help='input batch size')
    parser.add_argument('--isize', type=int, default=2, help='the size of the input to network')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=1000)
    parser.add_argument('--ndf', type=int, default=1000)
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00018, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00018, help='learning rate for Generator, default=0.00005')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='wgan-gp', help='Which model to use, default is gan')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--optim', type=str, default='rmsprop', help='Which optimizer to use, default is rmsprop')
    opt = parser.parse_args()
    writer = SummaryWriter()
    if opt.experiment is None:
        opt.experiment = 'samples/test'
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)

    real_data, data_loader = get_data(opt.dataroot, opt.batchSize)

    # 根据参数建立模型
    if opt.mode == 'wgan':
        netG = Generator(opt.nz, opt.ngf, opt.isize)
        netD = Discriminator(opt.isize, opt.ndf, wgan=True)
    elif opt.mode == 'wgan-gp':
        netG = Generator(opt.nz, opt.ngf, opt.isize)
        netD = Discriminator(opt.isize, opt.ndf, wgan=True)
    else:
        netG = Generator(opt.nz, opt.ngf, opt.isize)
        netD = Discriminator(opt.isize, opt.ndf)
    print(netG, '\n', netD)

    criterion = nn.BCELoss()
    if opt.cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        criterion = criterion.cuda()

    # 根据参数选择优化器
    if opt.optim == 'adam':
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.b1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.b1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    for epoch in range(opt.niter):
        netG.train()
        netD.train()
        lossG, lossD = 0.0, 0.0

        for i, data in enumerate(data_loader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            # 生成标签，只有gan用的到
            label_real = torch.ones((batch_size,)).view(-1, 1)
            label_fake = torch.zeros((batch_size,)).view(-1, 1)
            # 生成随机噪声
            noise = torch.randn(batch_size, opt.nz)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
                label_real = label_real.cuda()
                label_fake = label_fake.cuda()
                noise = noise.cuda()

            ############################
            # (1) Update D network
            ###########################
            if opt.mode == 'wgan':
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            netD.zero_grad()

            if opt.mode == 'wgan':
                # maximize D(x) - D(G(z)) <==> min D(G(z)) - D(x)
                errD_real = netD(real_cpu).mean()
                fake = netG(noise)
                errD_fake = netD(fake).mean()
                errD = -(errD_real - errD_fake)
                errD.backward()
            elif opt.mode == 'wgan-gp':
                # maximize D(x) - D(G(z)) + GP <==> min D(G(z)) - D(x) + GP
                errD_real = netD(real_cpu).mean()
                fake = netG(noise)
                errD_fake = netD(fake).mean()
                gradient_penalty = calc_gradient_penalty(netD, real_cpu.data, fake.data)
                errD = -(errD_real - errD_fake) + gradient_penalty * 0.1
                errD.backward()
            else:
                # maximize log(D(x)) + log(1 - D(G(z)))
                errD_real = criterion(netD(real_cpu), label_real)
                fake = netG(noise)
                errD_fake = criterion(netD(fake.detach()), label_fake)
                errD = errD_real + errD_fake
                errD.backward()
            lossD += errD.item()
            optimizerD.step()

            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()
            noise = torch.randn(batch_size, opt.nz)
            if opt.cuda:
                noise = noise.cuda()
            fake = netG(noise)
            if opt.mode == 'wgan':
                errG = - netD(fake).mean()
                errG.backward()
            elif opt.mode == 'wgan-gp':
                errG = -netD(fake).mean()
                errG.backward()
            else:
                # maximize log(D(G(z)))
                errG = criterion(netD(fake), label_real)
                errG.backward()
            lossG += errG.item()
            optimizerG.step()

        writer.add_scalar('train/G_loss', -lossG / len(data_loader), epoch + 1, walltime=epoch + 1)
        writer.add_scalar('train/D_loss', -lossD / len(data_loader), epoch + 1, walltime=epoch + 1)
        writer.close()

        print('[%d/%d] Loss_D: %.8f Loss_G: %.8f'
              % (epoch + 1, opt.niter, -lossD / len(data_loader), lossG / len(data_loader)))

        # 每5轮可视化一次
        if (epoch + 1) % 5 == 0:
            visualize(netG, netD, real_data, epoch + 1, opt.experiment, mode=opt.mode, cuda=opt.cuda)
