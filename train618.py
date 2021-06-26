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

### 该文件用来训练vgg618模型（model618） ，训练命令后面被注释掉了，直接load模型参数，因为训练好了要评估模型生成对应的混淆矩阵等

meldset = SpokenDigit(transforms.Compose([transforms.ToTensor()]))
size = len(meldset)
val_size = int(0.1 * size)
train_size = size - val_size 
train_dset, val_dset = random_split(meldset, [train_size, val_size])

train_dl = DataLoader(train_dset, 64, shuffle=True, num_workers=6, pin_memory=True)
val_dl = DataLoader(val_dset, 64, num_workers=6, pin_memory=True)

device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
##### test
# val_dset = SpokenDigit_test(transforms.Compose([transforms.ToTensor()]))
# val_dl = DataLoader(val_dset, 64, num_workers=6, pin_memory=True)
# device = get_default_device()
# val_dl = DeviceDataLoader(val_dl, device)

def evaluate(model, loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss" : loss.item(), "accuracy" : accuracy.item()}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(model, train_loader, val_loader, epochs, lr, optimizer_function = torch.optim.Adam):
    history = []
    optimizer = optimizer_function(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        print("Epoch ", epoch)
        #Train
        model.train()
        lrs = []
        tr_loss = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            tr_loss.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
        #Validate
        result = evaluate(model, val_loader)
        result["lrs"] = lrs
        result["train loss"] = torch.stack(tr_loss).mean().item()
        print("Last lr: ", lrs[-1]," Train_loss: ", result["train loss"], " Val_loss: ", result['loss'], " Accuracy: ", result['accuracy'])
        history.append(result)         
    torch.save(model.state_dict(),"./model/619_3.pt")
    return history
model = SpokenDigitModel()
# model = torch.load("./model/618_4")
# model.load_state_dict(torch.load("./model/619_1"))
model = to_device(model, device)

history = []
evaluate(model, val_dl)

torch.cuda.empty_cache()

history.append(fit(model, train_dl, val_dl, 128, 0.001))
# r = evaluate(model, val_dl)
# print("Loss: ", r['loss'], "\nAccuracy: ", r['accuracy'])
losses = []
accuracies = []
last_lr = []
for i in history[0]:
    losses.append(i['loss'])
    accuracies.append(i['accuracy'])
    last_lr.append(i['lrs'][-1])

def plot(var, title):
    plt.plot(var, '-x')
    plt.title(title)

plot(losses, 'Losses')
plt.savefig('./evalGraph/losses.jpg', dpi=500, bbox_inches='tight',pad_inches=0)
plt.close()
plot(accuracies, 'Accuracy')
plt.savefig('./evalGraph/accuracies.jpg', dpi=500, bbox_inches='tight',pad_inches=0)
plt.close()
plot(last_lr, 'Last Learning Rate')
plt.savefig('./evalGraph/lr.jpg', dpi=500, bbox_inches='tight',pad_inches=0)
plt.close()

