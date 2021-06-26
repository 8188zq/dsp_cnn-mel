import torch
import tarfile
from torchvision.datasets.utils import download_url
import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import csv
from PIL import Image
import matplotlib.pyplot as plt
from mfcc import *

def extract_mel(path):
    data, sr = librosa.load(path)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # data_path = path
    #win = 256
    # wlen = 256
    # nfft = wlen 
    # win = hanning_window(wlen)
    # inc = 128
    # wavedata,nframes,framerate,energy,zeroCrossingRate = read(data_path,win,inc)
    # F = point_check(wavedata,win,inc,nframes,framerate,energy,zeroCrossingRate)
    # # print("data: " , data.shape)
    # print("F: ", F.shape)
    # print(type(F))
    # F = F.flatten()
    # print("F: ", F.shape)
    S = librosa.feature.melspectrogram(y=data, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)
    # file  = './melgraph/' + str(basename[:-4]) + '.jpg'       #### melgraph是训练模型的时候用的
    file  = './melgraphtest/' + str(basename[:-4]) + '.jpg'    ### melgrahtest是测试用的
    plt.savefig(file, dpi=500, bbox_inches='tight',pad_inches=0)
    
    plt.close()
if __name__ == '__main__':
    templist = os.listdir('voices')
    voicelist = [os.path.join('voices/', i) for i in templist]
    # print(voicelist[99])
    # extract_mel(voicelist[99])
    for j in range(len(voicelist)):
        extract_mel(voicelist[j])
        print(str(j) + " is finished in toatal "+ str(len(voicelist)))
         