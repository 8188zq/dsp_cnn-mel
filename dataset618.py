import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
import torch
import os
import numpy as np
from fea_extraction import * 
import random
import matplotlib.pyplot as plt
import librosa
from PIL import Image

class SpokenDigit(Dataset):
    def __init__(self,transform = None):
        templist = os.listdir('melgraph')
        self.graphlist = [os.path.join('melgraph/', i) for i in templist]
        self.transform = transform

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, i):
        label = int(self.graphlist[i][-9:-7]) - 1
        ik = self.graphlist[i][:-4]+'.jpg'
        img = Image.open(ik)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

class SpokenDigit_test(Dataset):
    def __init__(self,transform = None):
        templist = os.listdir('melgraph_test')
        self.graphlist = [os.path.join('melgraph_test/', i) for i in templist]
        self.transform = transform

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, i):
        label = int(self.graphlist[i][-9:-7]) - 1
        ik = self.graphlist[i][:-4]+'.jpg'
        img = Image.open(ik)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)
    
##下面这个是之前的版本，已经弃用

# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self,train=True):
#         templist = os.listdir('Data')
#         # templist = os.listdir('Datatest')
#         if train:
#             self.voicelist = [os.path.join('Data/', i) for i in templist[:5440]]
#             # self.voicelist = [os.path.join('Datatest/', i) for i in templist[:5440]]
#         else:
#             self.voicelist = [os.path.join('Data/', i) for i in templist[5440:]]
#         # random.seed(20, 332)
#         # random.shuffle(templist)

#     def __getitem__(self, item):
#         # print(self.voicelist[item])
#         # temp = torch.Tensor(mel_spec(self.voicelist[item]))
#         # print(int(self.voicelist[item][-9:-7]) - 1)
#         transform1 = transforms.Compose([
#                 transforms.Resize((224,224)),
#                 transforms.ToTensor(),
#             ]
#         )
#         temp = mel_spec(self.voicelist[item])
#         temp_tensor = torch.Tensor(temp)
#         temp_tensor = temp_tensor.unsqueeze(0)
#         temp_img = transforms.ToPILImage()(temp_tensor)
#         temp_tensor1 = transform1(temp_img)
#         # print(temp_tensor1.shape)
#         return temp_tensor1, int(self.voicelist[item][-9:-7]) - 1

#     def __len__(self):
#         return len(self.voicelist)


### 这个是在测试
# s = MyDataset()
# # print(s.__len__())
# # for i in range(400):
# s.__getitem__(1)