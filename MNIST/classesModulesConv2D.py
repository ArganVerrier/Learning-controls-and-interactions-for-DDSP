# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 18:13:27 2020

@author: argan

Classe implémentée avec Conv2D (fonctionne)
"""
import matplotlib.pyplot as plt
plt.style.use('seabornCustom')
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distrib
import torchvision
import torch.nn.functional as f


import seaborn as sns

import multiprocessing

device=torch.cuda.current_device()
print("Current device : {}".format(torch.cuda.get_device_name(device)))

class AE(nn.Module):
    def __init__(self, encoder, decoder, encoding_dim):
        super(AE, self).__init__()
        self.encoding_dims = encoding_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(AE):

    def __init__(self, encoder, decoder, encoding_dims, latent_dims):
        super(VAE, self).__init__(encoder, decoder, encoding_dims)

        self.latent_dims = latent_dims

        #self.mu = nn.Sequential(nn.Linear(self.encoding_dims, self.latent_dims), nn.ReLU())
        self.linMu=nn.Linear(self.encoding_dims, self.latent_dims)
        #self.sigma = nn.Sequential(nn.Linear(self.encoding_dims, self.latent_dims), nn.Softplus())
        self.linSigma=nn.Linear(self.encoding_dims, self.latent_dims)

    def encode(self, x):

        x=self.encoder.forward(x)
        mu=f.relu(self.linMu(x))
        sigma=f.softplus(self.linSigma(x))
        # print("shape mu="+str(mu.shape))
        # print("sigma mu="+str(sigma.shape))

        return mu, sigma

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div

    def latent(self, x, z_params):

        n_batch=x.shape[0]
        mu, sigma =z_params
        #print(mu.shape)
        # print(mu.device)
        # print(sigma.device)
        #reparamétrisation
        q=distrib.Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        sampled=q.sample()
        sampled=sampled.cuda(device)
        # print(sampled.device)
        z=sigma*sampled + mu
        #compute KL divergence
        kl_div=0.5* torch.sum(1+sigma -torch.pow(mu, 2)- torch.exp(sigma))
        kl_div=kl_div/n_batch

        #print("espace latent, dim = "+str(z.shape))
        return z, kl_div


class encoder(nn.Module):
    def __init__(self, nin, n_latent = 2, n_hidden = 512, n_classes = 1):
        super(encoder, self).__init__()
        self.nin=nin
        self.n_latent=n_latent
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.lin1 = nn.Linear(int(nin/4), n_hidden)
        self.conv1= nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding=1)
        self.lin2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        #print(self.nin)
        im=self.conv1(x)
        #print(im.shape)
        im=f.max_pool2d(im, kernel_size=(2,2))
        #print(im.shape)
        im=im.view(-1, int(self.nin/4))
        #print(im.shape)
        a1=f.relu(self.lin1(im))
        a2=f.relu(self.lin2(a1))
        a3=f.relu(self.lin2(a2))

        return a3

class decoder(nn.Module):
    def __init__(self, nin, n_latent = 16, n_hidden = 512, n_classes = 1):
        super(decoder, self).__init__()
        self.nin=nin
        self.n_latent=n_latent
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.lin1=nn.Linear(n_latent, n_hidden)
        self.lin2=nn.Linear(n_hidden, n_hidden)
        self.lin3=nn.Linear(n_hidden, nin)

    def forward(self, x):
        x=x.view((-1, self.n_latent)).cuda(device)
        a1=f.relu(self.lin1(x))
        a2=f.relu(self.lin2(a1))
        a3=f.relu(self.lin3(a2))

        return a3

def construct_encoder_decoder_modules(nin, n_latent = 16, n_hidden = 512, n_classes = 1):
    encoder_module=encoder(nin=nin, n_latent=n_latent, n_hidden=n_hidden, n_classes=n_classes)
    decoder_module=decoder(nin=nin, n_latent=n_latent, n_hidden=n_hidden, n_classes=n_classes)

    return encoder_module, decoder_module


#%
recons_criterion = torch.nn.MSELoss(reduction='mean')

def compute_loss(model, x, batch_size):

    x_, kl_div=model.forward(x)
    #x_.view(x.shape)
    pxz=recons_criterion(x_.view(x.shape), x)
    full_loss=torch.mean(torch.log(pxz))-kl_div


    return full_loss, x_

def train_step(model, x, optimizer, batch_size):
    loss, x_=compute_loss(model, x, batch_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, x_

def predIm(x, y, model):
    coord=torch.tensor([x, y]).cuda()
    pred=model.decode(coord).detach().reshape((28,28))
    return pred
