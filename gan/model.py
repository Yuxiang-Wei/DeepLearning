import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, isize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, isize),
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, isize, ndf, wgan=False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.wgan = wgan
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        if not self.wgan:
            output = self.sigmod(output)
        else:
            output = output.mean(0)
        return output
