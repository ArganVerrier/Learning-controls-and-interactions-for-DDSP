# Learning-controls-and-interactions-for-DDSP

## 1. VAE MNIST/FashionMNIST

We trained 2 VAEs on MNIST and FashionMNIST Datasets. They are respectively collection of digits and clothes (t-shirts, sneekers, Sweetshirts, etc..). 
There are 2 Notebooks, one for each Datasets. We tested Linear architecture, and then saw that Convolution architecture is even more efficient for Images Problems. 

You can either train you model on the Notebooks (GPU prefered, Nvidia RTX personal recommandation... hihihi), or load a pre-trained model called "model_MNIST.pt" and "model_FashionMNIST.pt" and test it. 

The model has achieved a reasonable reconstruction quality. We also generate new data by sampling and interpolation in its latent space. Everything can be tested in the Notebooks 
"MNIST_VAE_Conv2d.ipynb"  and "FashionMNIST_VAE_Conv2d.ipynb"

## 2. Solordinario

In this part, we used the dataset 'Solordinario' witch is a library made by Ircam, composed of single notes from a violon recorded in mono. This a paid and all right reserved dataset, so, we are note allowed to share it, but all the results of our training are available.

We used this dataset to create a dictionnary of pitch and loudness envelopes at a fram rate oh 250Hz. With this envelopes, we have trained a VAE to encode in a latent space of 64 dimensions and be able to reconstruct those envelopes with a decoder. Using a pretrained DDSP, we are finally able to reconstruct audio file from the envelopes.

![Screenshot](Schema.png)

By interpolation in the latent space, it is then possible to create new audio files for audio synthesis. 

There is a readme.txt in the file "Solordinario" that explainded more precisely how to use our models and codes.
