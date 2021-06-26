import torchvision
from torchvision import transforms
import torch
import os
import numpy as np
from fea_extraction import * 
import random
import matplotlib.pyplot as plt
import librosa

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        # templist = os.listdir('Data')
        templist = os.listdir('Datatest')
        
        # self.voicelist = [os.path.join('Data/', i) for i in templist]
        self.voicelist = [os.path.join('Datatest/', i) for i in templist]
        
        # random.seed(20, 332)
        # random.shuffle(templist)

    def __getitem__(self, item):
        # print(self.voicelist[item])
        temp = mel_spec(self.voicelist[item])
        temp1 = torch.Tensor(temp)
        # print(int(self.voicelist[item][-9:-7]) - 1)
        # transform1 = transforms.Compose([
        #         transforms.Resize((128,98)),
        #         transforms.ToTensor(),
        #     ]
        # )
        # temp = mel_spec(self.voicelist[item])
        # temp_tensor = torch.Tensor(temp)
        # temp_tensor = temp_tensor.unsqueeze(0)
        # temp_img = transforms.ToPILImage()(temp_tensor)
        # temp_tensor1 = transform1(temp_img)
        # print(temp_tensor1.shape)
        # return temp_tensor1, int(self.voicelist[item][-9:-7]) - 1
        ###解释为什么用dic：因为这个之前用前面几届的数据预训练，他们的预测的东西和这次有所区别，有的是乱序，有的是没出现，我为了匹配，就有了dic这个映射表，
        ###而cid是用来解码的
        dic = {1: 0, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 12, 11: 13, 12: 14, 15: 16, 16: 19, 17: 15, 18: 17,
               19: 18, 0: 1, 2: 3, 13: 10, 14: 11}     
        return temp1.view(1,temp.shape[0],-1), dic[int(self.voicelist[item][-9:-7]) - 1]

    def __len__(self):
        return len(self.voicelist)

# s = MyDataset()
# # print(s.__len__())
# # for i in range(400):
# s.__getitem__(1)