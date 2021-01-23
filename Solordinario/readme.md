# Description of the files

#### Solordinario_PrepareDataSet.ipynb
Notebook to prepare the dataset : 
    - Get only the 3 first secondes of all files to homogenize the length of all file
    - Compute loudness and pitch envelopes with function in the file "descriptors.py"
    - Store all enveloppes in a dictionary "data_dict.npy"
    
Organisation of "data_dict.npy" :

    - dict_dict['env'] => array (2*750) with loudness and pitch envelopes 
    
    - dict_dict['file_name'] => Original file name 
    
    - dict_dict['vel'] => Velocity ('ff' = forte ; 'mf' = mezzo forte ; 'pp' = piano) 
    
    - dict_dict['pitch'] => Pitch ('A5', 'B#6', ...)  
    
    - dict_dict['corde'] => String Played ('1c', '2c', '3c', '4c') 
    
    
#### VAE_Solordinario_VF.ipynb
Notebook to trained the model to encode envelopes in a latent space of 64 dimensions and reconstruct them with a decodeur. The model trained is save in the file "vae_model_trained".
To visualise and listen the reconstructions, we have create a class in file "visualizer.py.". 

#### Visualizer_MANUAL.ipynb
Notebook that explained and show how the visualizer.py works.

#### VAE_Solordinario_ExploreLatentSpace.ipynb
Notebook used to compute new envelopes by interpolation in the latent space. 
We have also tried to understand how the VAE organised the latent space. The visualisation is in 2 dimensions. The dimension reduction is made using PCA and TSNE.
