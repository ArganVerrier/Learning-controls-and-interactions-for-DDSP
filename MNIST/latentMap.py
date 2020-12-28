# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:27:06 2020

@author: argan

Un essai de plot de l'espace latent. Pour chaque image on récupère le label et les moyennes des données encodées, puis on plot.

"""
import matplotlib.pyplot as plt
plt.style.use('seabornCustom')
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distrib
import torchvision

from classesModulesConv2D import *
import seaborn as sns
device=torch.cuda.current_device()
import multiprocessing
PATH="D:/Documents/Cours/M2/projet-ml/Learning-controls-and-interactions-for-DDSP/testModelcuda"
dataset_dir = 'D:\Documents\Cours\M2\projet-ml\Learning-controls-and-interactions-for-DDSP\data'
dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)

imRef, _=next(iter(dataset))

nin=imRef.shape[1]*imRef.shape[2]

nHidden=512

nlatent=2

encoder, decoder=construct_encoder_decoder_modules(nin, n_latent = 2, n_hidden = 512, n_classes = 1)

model=VAE(encoder.cuda(), decoder.cuda(), nHidden, latent_dims=nlatent)

model.load_state_dict(torch.load(PATH))
model.eval()

model=model.cuda()

Nplot=500
iterator=iter(dataset)

listRes=[list(([],[])) for ind in range(10)]

for ind in range(Nplot):
    print(ind)
    image, label=next(iterator)
    image=image.cuda()
    z_param=model.encode(image.unsqueeze(0))
    #coord=model.latent(image, z_param)
    mu=z_param[0].detach()
    listRes[label][0].append(mu[0,0].item())
    listRes[label][1].append(mu[0,1].item())

fig=plt.figure()
for ind in range(len(listRes)):
    plt.plot(listRes[ind][0], listRes[ind][1], "*", label=str(ind))
    
x = np.linspace(-1.1, 1.1, 8, dtype=float)
y = np.linspace(-0.1, 0.1, 8, dtype=float)
fig = plt.figure(figsize=(10, 8))
for i in range(8):
    for j in range(8):
        plt.figure()
        x_=float(x[i])
        y_=float(y[j])
        pred=predIm(x_, y_, model)
        plt.imshow(pred.cpu())
