import torch
import numpy as np
import torch.nn as nn
# from tensorboardX import SummaryWriter
import argparse
import torch.utils.data as Data

from model import vgg11_bn
from dataset1 import *
from tqdm import tqdm
from Model import DSPClassify


# xwriter = SummaryWriter('cnn_melspec_log')
# data_feed = DataFeed()
cates = "数字 语音 语言 识别 中国 忠告 北京 背景 上海 商行 Speech Speaker Signal Sequence Process Print Project File Open Close".split(' ')
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
cid = {0: 1, 2: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 13: 11, 14: 12, 16: 15, 19: 16, 15: 17,
           17: 18, 18: 19, 1: 0, 3: 2, 10: 13, 11: 14}



def train3(model,epoch_num=36):
    mtd = MyDataset()
    MyTrainDataloader = Data.DataLoader(dataset=mtd, batch_size=32, shuffle=True, drop_last=True)
    model.train()
    model.cuda()
    tbar = tqdm(range(10))
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().cuda()
    for iter in tbar:
        losses = []
        for data, cls in tqdm(MyTrainDataloader):
            data = data.cuda()
            cls = cls.cuda()
            optimizer.zero_grad()
            predicted = model(data)
            loss = criterion(predicted, cls.long())
            loss.backward()
            optimizer.step()
            tbar.set_postfix(loss=loss.mean().item())
            losses.append(loss.data.item())
            #print(loss)
        losses = np.array(losses)
        print("epoch loss:")
        print(np.mean(losses))
    torch.save(model, './model/dsp1.pt')


#############################################################################

def train2(model,dataloader,epoch_num=100):
    model.train()
    model.cuda()
    criterian = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(epoch_num):
        losses = []
        for i,(data,label) in tqdm(enumerate(dataloader)):
            # print(data.shape)
            data= data.cuda()
            label = label.cuda()
            predict = model(data)
            loss = criterian(predict,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())
            print(loss)
        # print("epoch "+str(epoch)+ " : {}".format(np.array(losses).mean()))
        losses = np.array(losses)
        print("epoch loss:")
        print(np.mean(losses))
    torch.save(model,"./model/3")
            
def predict(model,dataloader,epoch_num=100):
    model.eval()
    model.cuda()
    for epoch in range(epoch_num):
        for i,(data,label) in tqdm(enumerate(dataloader)):
            data = data.cuda()
            label = label.cuda()
            y = model(data)
            predict = torch.tensor([cid[i.item()] for i in torch.argmax(y,dim=1)])
            predict = predict.cuda()
            print(predict.shape)
            print(label.shape)
            print(predict - label)

    ###解释为什么用dic：因为这个之前用前面几届的数据预训练，他们的预测的东西和这次有所区别，有的是乱序，有的是没出现，我为了匹配，就有了dic这个映射表，
    ###而cid是用来解码的       
cid = {0: 1, 2: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 13: 11, 14: 12, 16: 15, 19: 16, 15: 17,
           17: 18, 18: 19, 1: 0, 3: 2, 10: 13, 11: 14}

def test_model(model_path, vgg=True):
    cm = np.zeros(shape=(20,20), dtype=np.int32)
    if vgg:
        model = vgg11_bn()
    else:
        model = DSPClassify()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    mtd = MyDataset()
    for _ in range(len(mtd)):
        data, cls = mtd[_]
        data = data.view(1, 1, 128, 98)
        predict = model(data)
        cm[cls, torch.argmax(predict).item()] += 1
    # plot_confusion_matrix(cm,labels)
    print(cm)

def plot_confusion_matrix(cm,labels, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./evalGraph/confusion_matrix.jpg')
    # plt.show()

if __name__ == "__main__":

    # dataset = MyDataset()
    # dataloader = Data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=2)

    # model = vgg11_bn()
    # model = torch.load("./model/dsp1.pt")

    # model = DSPClassify()
    # model.load_state_dict(torch.load("./model/DSP.pt"))
    # test_model("./model/DSP.pt",vgg=False)
    test_model("./model/VGG.pt",vgg=True)
    #train2(model,dataloader)
    # train3(model)

    # predict(model,dataloader)