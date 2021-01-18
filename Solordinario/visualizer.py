import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sounddevice as sd
import numpy as np
import torch


class Visualizer:
    def __init__(self, list_images, list_sources = None ):
        super().__init__()
        self.list_images = list_images
        self.list_sources = list_sources
        self.list_sounds = None

        self.compare = self.list_sources!=None
        if self.compare and len(self.list_sources)<len(self.list_images):
            raise Exception("The source list must be the same length as image list")


    def show_pitch(self,indexes=None): # index = tuple (a,b)
        """Plot pitches of images included in indexes [a;b]"""
        if indexes==None: 
            a,b = 0,len(self.list_images)-1
        else: 
            a,b = indexes
        number_images = b-a+1
        fig, axs = plt.subplots(int(np.ceil(number_images/3)),3, figsize=(20, 4*int(np.ceil(number_images/3))), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.2)
        axs = axs.ravel()
        for j in range(number_images):
            y = self.list_images[j][1]
            if self.compare:
                ys = self.list_sources[j][1]
                axs[j].plot(ys,label="source")
            axs[j].plot(y,label="VAE")
            axs[j].set_title("Pitch of sample n°{}".format(a+j))
            axs[j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
            axs[j].legend()

    def show_loudness(self,indexes=None): # index = tuple (a,b)
        """Plot Loudness of images included in indexes [a;b]"""
        if indexes==None: 
            a,b = 0,len(self.list_images)-1
        else: 
            a,b = indexes
        number_images = b-a+1
        fig, axs = plt.subplots(int(np.ceil(number_images/3)),3, figsize=(20, 4*int(np.ceil(number_images/3))), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.1)
        axs = axs.ravel()
        for j in range(number_images):
            y = self.list_images[j][0]
            if self.compare:
                ys = self.list_sources[j][0]
                axs[j].plot(ys,label="source") 
            axs[j].plot(y,label="VAE")           
            axs[j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
            axs[j].set_title("Loudness of sample n°{}".format(a+j))
            axs[j].legend()    

    def reconstruction(self,ddsp,loudness_max = 1,pitch_max = 1):
        self.list_sounds = []
        for image in self.list_images:
            pitch = torch.from_numpy(image[1]*pitch_max).float().reshape(1, -1, 1)
            loudness = torch.from_numpy(image[0]*loudness_max).float().reshape(1, -1, 1)
            self.list_sounds.append(ddsp(pitch, loudness).squeeze().detach().numpy())


    def show_sound(self, indexes=None, Fs = 16000):
        """Plot soundwave of images included in indexes [a;b]"""
        if indexes==None: 
            a,b = 0,len(self.list_sounds)-1
        else: 
            a,b = indexes
        number_sounds = b-a+1
        fig, axs = plt.subplots(int(np.ceil(number_sounds/3)),3, figsize=(20, 4*int(np.ceil(number_sounds/3))), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.5)
        axs = axs.ravel()
        for j in range(number_sounds):
            y = self.list_sounds[j]
            x = np.array([i/Fs for i in range(len(y))])
            axs[j].plot(x,y, label="s(t)")
            axs[j].set_title("Soundwave of sample n°{}".format(a+j))
            axs[j].set_xlabel("time (s)")
            axs[j].set_ylabel("A")
            axs[j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
            axs[j].legend()    



    def show_spectrogramm(self, indexes = None, Nfft = 256, Fs = 16000):
        """Plot spectrogram of sounds included in indexes [a;b]"""
        if indexes==None: 
            a,b = 0,len(self.list_sounds)-1
        else: 
            a,b = indexes

        number_sounds = b-a+1
        fig, axs = plt.subplots(int(np.ceil(number_sounds/3)),3, figsize=(20, 4*int(np.ceil(number_sounds/3))), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.3)
        axs = axs.ravel()
        for j in range(number_sounds):
            sound = self.list_sounds[j]
            axs[j].specgram(sound, NFFT=Nfft, Fs=Fs, noverlap=Nfft/2)
            axs[j].set_title("Spectrogram of sample n°{}".format(a+j))
            axs[j].set_xlabel("time (s)")
            axs[j].set_ylabel("freq (Hz)")
            axs[j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
            axs[j].legend()    

    def listen(self, index=0):
        sig = self.list_sounds[index]
        sd.play(sig*0.5/np.max(sig), 16000)
