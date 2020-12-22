# Learning-controls-and-interactions-for-DDSP

We trained 2 VAEs on MNIST and FashionMNIST Datasets. They are respectively collection of digits and clothes (t-shirts, sneekers, Sweetshirts, etc..). 
There are 2 Notebooks, one for each Datasets. We tested Linear architecture, and then saw that Convolution architecture is even more efficient for Images Problems. 

You can either train you model on the Notebooks (GPU prefered, Nvidia RTX personally recommand... hihihi), or load a pre-trained model called "model_MNIST.pt" and "model_FashionMNIST.pt" and test it. 

The model has achieved a reasonable reconstruction quality. We also generate new data by sampling and interpolation in its latent space. Everything can be tested in the Notebooks 
"MNIST_VAE_Conv2d.ipynb"  and "FashionMNIST_VAE_Conv2d.ipynb"







