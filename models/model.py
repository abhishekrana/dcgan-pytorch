import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

class Model():
    def __init__(self, config):

        self.config = config
        self.device = self.config['device']
        self.batch_size = config['data_handler']['batch_size']
        self.ngpu = self.config['ngpu']
        self.nz = self.config['nz']

        # Generator
        self.netG = Generator(self.config).to(self.device)
        self.netG.apply(weights_init)
        if config['netG'] != '':
            self.netG.load_state_dict(torch.load(self.config['netG']))
        logging.debug(self.netG)

        # Discriminator
        self.netD = Discriminator(self.config).to(self.device)
        self.netD.apply(weights_init)
        if config['netD'] != '':
            self.netD.load_state_dict(torch.load(self.config['netD']))
        logging.debug(self.netD)

        ## Loss
        self.criterion = nn.BCELoss()

        ## Optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config['optimizer']['lr'], 
                betas=(self.config['optimizer']['beta1'], 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config['optimizer']['lr'], 
                betas=(self.config['optimizer']['beta1'], 0.999))


# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.ngpu = self.config['ngpu']
        self.nz = self.config['nz']
        self.ngf = self.config['ngf']
        self.nc = self.config['nc']

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,     self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.ConvTranspose2d(    self.ngf,      self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.ngpu = self.config['ngpu']
        self.ndf = self.config['ndf']
        self.nc = self.config['nc']
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

