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


class PointDataset(Dataset):
    def __init__(self, data):
        self.input = data

    def __getitem__(self, index):
        return torch.from_numpy(self.input[index])

    def __len__(self):
        return len(self.input)


def get_data(data_root, batch_size):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./', help='path to dataset')
    parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
    parser.add_argument('--isize', type=int, default=2, help='the size of the input to network')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=1000)
    parser.add_argument('--ndf', type=int, default=1000)
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00018, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00018, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='wgan-gp', help='Which model to use, default is gan')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--optim', type=str, default='rmsprop', help='Which optimizer to use, default is rmsprop')
    opt = parser.parse_args()

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    original_data, data_loader = get_data(opt.dataroot, opt.batchSize)
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
    one = torch.FloatTensor([1])
    mone = one * -1
    if opt.cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        criterion = criterion.cuda()
        one = one.cuda()
        mone = mone.cuda()

    if opt.optim == 'adam':
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    for epoch in range(opt.niter):
        outputs = []
        for i, data in enumerate(data_loader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            label_real = torch.ones((batch_size,))
            label_fake = torch.zeros((batch_size,))
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
                errD_real = netD(real_cpu)
                errD_real.backward(one)
                fake = netG(noise)
                errD_fake = netD(fake)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
            elif opt.mode == 'wgan-gp':
                errD_real = netD(real_cpu)
                errD_real.backward(one)
                fake = netG(noise)
                errD_fake = netD(fake)
                errD_fake.backward(mone)
                gradient_penalty = calc_gradient_penalty(netD, real_cpu.data, fake.data)
                gradient_penalty.backward()
                errD = errD_real - errD_fake + gradient_penalty * 0.1
            else:
                # maximize log(D(x)) + log(1 - D(G(z)))
                errD_real = criterion(netD(real_cpu), label_real)
                fake = netG(noise)
                errD_fake = criterion(netD(fake.detach()), label_fake)
                errD = errD_real + errD_fake
                errD.backward()
            outputs.append(fake.data)
            optimizerD.step()

            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()
            noise = torch.randn(batch_size, opt.nz)
            fake = netG(noise)
            if opt.cuda:
                noise = noise.cuda()
            if opt.mode == 'wgan':
                errG = netD(fake)
                errG.backward(one)
            elif opt.mode == 'wgan-gp':
                errG = netD(fake)
                errG = errG.mean()
                errG.backward(one)
            else:
                # maximize log(D(G(z)))
                errG = criterion(netD(fake), label_real)
                errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(data_loader),
                     errD.item(), errG.item()))
        plt.scatter(original_data[:, 0], original_data[:, 1], c='b')
        for fake in outputs:
            x = fake.numpy()
            plt.scatter(x[:, 0], x[:, 1], c='r')
        plt.show()
