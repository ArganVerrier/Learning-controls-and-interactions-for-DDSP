# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:31:18 2020

@author: argan
"""
import matplotlib.pyplot as plt
plt.style.use('seabornCustom')
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distrib
import torchvision
from scipy.special import logsumexp
from classesModulesConv2D import *
import seaborn as sns

import multiprocessing


if __name__ == "__main__":
    #%Train/valid data set
    dataset_dir = 'D:\Documents\Cours\M2\projet-ml\Learning-controls-and-interactions-for-DDSP\data'
    
    # Going to use 80%/20% split for train/valid
    valid_ratio = 0.2
    
    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid =  int(valid_ratio * len(train_valid_dataset))
    
    # Load the test set
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])
    test_dataset = torchvision.datasets.MNIST(root=dataset_dir, transform=torchvision.transforms.ToTensor(),train=False)
    
    # Prepare 
    num_threads = 4     # Loading the dataset is using 4 CPU threads
    batch_size  = 128   # Using minibatches of 128 samples
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)
    torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_threads)
        
    #% training
    imRef, labRef=next(iter(train_loader))
    
    nClass=10
    nHidden=512
    nIn=imRef[1, 0, :, :].shape[0]*imRef[1, 0, :, :].shape[1]
    nOut=nIn*nClass
    
    Nepoch=50
    
    encoder, decoder=construct_encoder_decoder_modules(nIn, n_latent=2, n_hidden=nHidden, n_classes=nClass)
    
    model=VAE(encoder, decoder, nHidden, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    losses=np.zeros((Nepoch,))


#%%

   
    for ind in range(Nepoch):
        print(ind)
        loss=0.
        for i,(img, lab) in enumerate(train_loader):
            lossStep, x_=train_step(model, img, optimizer, batch_size)
            loss+=lossStep
        losses[ind]=loss.mean().item()
    
#%%
    # x=np.linspace(-8., 8., 3)
    # y=np.linspace(-8.,8.,3)
    # fig=plt.figure()
    # for indx in range(x.size):
    #     x_=x[indx]
    #     for indy in range(y.size):
    #         y_=y[indy]
    #         ax=fig.add_subplot(3,3, indy+indx+1)
    #         plt.imshow(predIm(x_, y_, model))
            
            
    
    # def evaluate_nll_bpd(data_loader, model, batch = 500, R = 5):
    # # Set of likelihood tests
    #     likelihood_test = []
    #     # Go through dataset
    #     for batch_idx, (x, _) in enumerate(data_loader):
    #         for j in range(x.shape[0]):
    #             a = []
    #             for r in range(0, R):
    #                 cur_x = x[j].unsqueeze(0)
    #                 # Repeat it as batch
    #                 x = cur_x.expand(batch, *cur_x.size()[1:]).contiguous()
    #                 x = x.view(batch, -1)
    #                 x_tilde, kl_div = model(x)
    #                 rec = reconstruction_loss(x_tilde, x, average=False)
    #                 a_tmp = (rec + kl_div)
    #                 a.append(- a_tmp.cpu().data.numpy())
    #             # calculate max
    #             a = np.asarray(a)
    #             a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
    #             likelihood_x = logsumexp(a)
    #             likelihood_test.append(likelihood_x - np.log(len(a)))
    #     likelihood_test = np.array(likelihood_test)
    #     nll = - np.mean(likelihood_test)
    #     # Compute the bits per dim (but irrelevant for binary data)
    #     bpd = nll / (np.prod(nin) * np.log(2.))
    #     return nll, bpd
    
    
    # x = np.linspace(-3, 3, 8)
    # y = np.linspace(-3, 3, 8)
    # fig = plt.figure(figsize=(10, 8))
    # for i in range(8):
    #     for j in range(8):
    #         plt.subplot(8, 8, (i * 8) + j + 1)
    #         final_tensor = torch.zeros(2)
    #         final_tensor[0] = x[i]
    #         final_tensor[1] = y[j]
    #         plt.imshow(model.decode(final_tensor).detach().reshape(28, 28), cmap='gray')
    #         plt.axis('off')
                
            