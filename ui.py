import torch
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
import torchvision
from torchvision.datasets.utils import download_url
import torch.nn as nn
import torch.nn.functional as F
import tarfile
import os
import librosa
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import matplotlib
import csv
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

from device import*
import device 
from model618 import *
from dataset618 import*
from fea_extraction618 import *
from model import *
from Model import *

### 这个是一个简单的展示效果代码，要求按test_01_xx格式将对应的音频文件放于voices文件下，然后先用fea_extraction618对其预处理
###（因为model618的输入是图，需要先把语音信号转成mel谱图信号）然后运行test会出预测的结果，p1+p2是vgg与dsp模型跑出的结果的综合结果，p3是model618抛出的结果，answer是综合后最终预测

def test():
    
    model2 = DSPClassify()
    model2.load_state_dict(torch.load('./model/DSP.pt'))
    model = vgg11_bn()
    model.load_state_dict(torch.load('./model/VGG.pt'))
    model3 = SpokenDigitModel()
    model3.load_state_dict(torch.load('./model/619_3.pt'))
    cates = "数字 语音 语言 识别 中国 忠告 北京 背景 上海 商行 Speech Speaker Signal Sequence Process Print Project File Open Close".split(' ')
    cid = {0: 1, 2: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 13: 11, 14: 12, 16: 15, 19: 16, 15: 17,
           17: 18, 18: 19, 1: 0, 3: 2, 10: 13, 11: 14}
    dic = {1: 0, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 12, 11: 13, 12: 14, 15: 16, 16: 19, 17: 15, 18: 17,
               19: 18, 0: 1, 2: 3, 13: 10, 14: 11}
    # data = torch.tensor(mel_spec('voices/test_01_00.wav')).view(1, 1, 128, 98)

    #### 三个model合并
    num = 0 
    for i in range(20):
        data = torch.tensor(mel_spec('voices/test_01_{0:02d}.wav'.format(i))).view(1, 1, 128, 98)
        print('voices/test_01_{0:02d}.wav'.format(i))
        predicted = model(data)
        predicted2 = model2(data)
        transform = transforms.Compose([transforms.ToTensor()])
        ik = 'melgraphtest/test_01_{0:02d}.jpg'.format(i)
        img = Image.open(ik)
        img = transform(img).view(1,3,385,387)
        print(ik)
        predicted3 = model3(img)
        p = predicted2 + predicted
        # print("p",p)
        p = F.normalize(p,dim=1)
        # predicted3 = F.normalize(predicted3,dim=1)
        # print("p_normailze",p)
        print("p1+p2: ",cates[cid[torch.argmax(p).item()]])
        for j in range(20):
            p[0][j] += predicted3[0][cid[j]]
            # print(j," : ",predicted3[0][dic[j]])
        print("p3: ",cates[torch.argmax(predicted3).item()])
        # print(p)
        # print(predicted3)
        print("answer :  ",cates[cid[torch.argmax(p).item()]])
        if cid[torch.argmax(p).item()] == i: num += 1
        print("i : ",i," ,pre : ",cid[torch.argmax(p).item()])
    print("accurate_num",num)


    
    #######  下面是model分开测
    '''
    for i in range(20):
        data = torch.tensor(mel_spec('voices/test_01_{0:02d}.wav'.format(i))).view(1, 1, 128, 98)
        print('voices/test_01_{0:02d}.wav'.format(i))
        predicted = model(data)
        predicted2 = model2(data)
        p = predicted2 + predicted
        print(cates[cid[torch.argmax(p).item()]])


    templist = os.listdir('melgraphtest')
    graphlist = [os.path.join('./melgraphtest/', i) for i in templist]
    transform = transforms.Compose([transforms.ToTensor()])
    for ik in graphlist:
        img = Image.open(ik)
        img = transform(img).view(1,3,385,387)
        print(ik)
        predicted3 = model3(img)
        p = predicted3
        # print(cates[cid[torch.argmax(p).item()]])
        print(cates[torch.argmax(p).item()])
    '''
test()