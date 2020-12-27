import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms




class Encoder(nn.Module):
    def __init__(self, latent_dims, capacity = 16):
        super(Encoder, self).__init__()
        self.latent_dims  = latent_dims
        self.capacity = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.capacity, kernel_size=1, stride=1, padding=0) # out: c, 2, 750
        self.conv2 = nn.Conv2d(in_channels=self.capacity, out_channels=self.capacity*2, kernel_size=1, stride=1, padding=0) # out: c*2, 2, 750
        self.fc_mu = nn.Linear(in_features=2*self.capacity*2*750, out_features=self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=2*self.capacity*2*750, out_features=self.latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims, capacity = 16):
        super(Decoder, self).__init__()
        self.latent_dims  = latent_dims
        self.capacity = capacity
        self.fc = nn.Linear(in_features=self.latent_dims, out_features=2*self.capacity*2*750)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.capacity*2, out_channels=self.capacity, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.capacity, out_channels=1, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*2, 2,750) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, capacity = 16):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.capacity = capacity
        self.encoder = Encoder(self.latent_dims,self.capacity)
        self.decoder = Decoder(self.latent_dims,self.capacity)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu