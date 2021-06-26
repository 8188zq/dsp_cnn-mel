import wave
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct
 

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    #print("分帧的shape：")
    #print(frameout.shape)
    #print(nwin)
    #print(inc)
    #print(nf)
    return frameout

#加窗
def hanning_window(N):
    nn = [i for i in range(N)]
    return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))

def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0

# 计算每一帧的能量 256个采样点为一帧
def calEnergy(wave_data,win,inc) :
    fdata = enframe(wave_data,win, inc)
    energy = []
    sum = 0
    x1 = fdata.shape[0]
    x2 = fdata.shape[1]
    for i in range(x1) :
        for j in range(x2) :
            sum = sum + (int(fdata[i][j]) * int(fdata[i][j]))
        energy.append(sum)
        sum = 0
    return energy

#计算过零率
def calZeroCrossingRate(wave_data,win,inc) :
    fdata = enframe(wave_data, win, inc)
    zeroCrossingRate = []
    sum = 0
    x1 = fdata.shape[0]
    x2 = fdata.shape[1]
    for i in range(x1) :
        for j in range(x2) :
            if j == 0 :
                continue
            sum = sum + np.abs(sgn(fdata[i][j]) - sgn(fdata[i][j - 1]))/2
        zeroCrossingRate.append(float(sum) / x2)
        sum = 0
    return zeroCrossingRate

def read(data_path,win,inc):
 '''读取语音信号
 '''
 wavepath = data_path
 f = wave.open(wavepath,'rb')
 params = f.getparams()
 nchannels,sampwidth,framerate,nframes = params[:4] #声道数、量化位数、采样频率、采样点数
 str_data = f.readframes(nframes) #读取音频，字符串格式
 f.close()
 wavedata = np.fromstring(str_data,dtype = np.short) #将字符串转化为浮点型数据
 energy = calEnergy(wavedata,win,inc)
 zeroCrossingRate = calZeroCrossingRate(wavedata,win,inc)
 wavedata = wavedata * 1.0 / (max(abs(wavedata))) #wave幅值归一化
 return wavedata,nframes,framerate,energy,zeroCrossingRate


def endPointDetect(wave_data, energy, zeroCrossingRate) :
    sum = 0
    energyAverage = 0
    for en in energy :
        sum = sum + en
    energyAverage = sum / len(energy)

    sum = 0
    for en in energy[:5] :
        sum = sum + en
    ML = sum / 5                        
    MH = energyAverage / 4             #较高的能量阈值
    ML = (ML + MH) / 4    #较低的能量阈值   
    ###  我把ML从4改成了8
    sum = 0
    for zcr in zeroCrossingRate[:5] :
        sum = float(sum) + zcr             
    Zs = sum / 5                     #过零率阈值

    A = []
    B = []
    C = []

    # 首先利用较大能量阈值 MH 进行初步检测
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH :
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1
        
        if flag == 1 and energy[i] < MH :
            A.append(i)
            flag = 0
    # print("较高能量阈值，计算后的浊音A:" + str(A))

    # 利用较小能量阈值 ML 进行第二步能量检测
    for j in range(len(A)) :
        i = A[j]
        if j % 2 == 1 :
            while i < len(energy) and energy[i] > ML :
                i = i + 1
            B.append(i)
        else :
            while i > 0 and energy[i] > ML :
                i = i - 1
            B.append(i)
    # print("较低能量阈值，增加一段语言B:" + str(B))

    # 利用过零率进行最后一步检测
    for j in range(len(B)) :
        i = B[j]
        if j % 2 == 1 :
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >=  2 * Zs :
                i = i + 1
            C.append(i)
        else :
            while i > 0 and zeroCrossingRate[i] >=  3 * Zs :
                i = i - 1
            C.append(i)
    # print("过零率阈值，最终语音分段C:" + str(C))
    return C

#语音信号端点检测
def point_check(wavedata,win,inc,nframes,framerate,energy,zeroCrossingRate):
 #注意：wavedata这里是一维的array，并没有转成1*n的矩阵   
 #1.计算短时过零率
 FrameTemp1 = enframe(wavedata,win,inc)
 N = endPointDetect(wavedata, energy, zeroCrossingRate)
# 画张图看看
#  print("N: ",N)
 StartPoint = N[0]
 if(len(N)==1):
     EndPoint = len(FrameTemp1)-1
 else:
    EndPoint = N[1]
 time = np.arange(0,nframes) * (1.0 / framerate)
 time= np.reshape(time,[nframes,1]).T
#  wavedata = np.reshape(wavedata,[nframes,1]).T
#  plt.plot(time[0,:nframes],wavedata[0,:nframes],c="b")
#  plt.xlabel("time")
#  plt.ylabel("amplitude")
#  plt.title("shop")
#  plt.vlines(StartPoint*inc* (1.0 / framerate), -1.0, 1.0, colors = "c", linestyles = "dashed")
#  plt.vlines(EndPoint*inc* (1.0 / framerate), -1.0, 1.0, colors = "c", linestyles = "dashed")
#  #plt.show()
#  plt.savefig('shop.png')
 return FrameTemp1[StartPoint:EndPoint]
 
 
def mfcc(FrameK,framerate,win):
 '''提取mfcc参数 
 input:FrameK(二维array):二维分帧语音信号，每帧256长
   framerate:语音采样频率
   win:分帧窗长（FFT点数）也就是wlen
 '''
 #mel滤波器
 mel_bank,w2 = mel_filter(24,win,framerate,0,0.5)
 FrameK = FrameK.T
 #计算功率谱
 S = abs(np.fft.fft(FrameK,axis = 0)) ** 2
 #将功率谱通过滤波器
 P = np.dot(mel_bank,S[0:w2,:])
 #取对数
 logP = np.log(P)
 #计算DCT系数
# rDCT = 12
# cDCT = 24
# dctcoef = []
# for i in range(1,rDCT+1):
#  tmp = [np.cos((2*j+1)*i*math.pi*1.0/(2.0*cDCT)) for j in range(cDCT)]
#  dctcoef.append(tmp)
# #取对数后做余弦变换 
# D = np.dot(dctcoef,logP)
 num_ceps = 12
 print("logP:")
 print(logP.shape)

 D = dct(logP,type = 2,axis = 0,norm = 'ortho')[1:(num_ceps+1),:]
 return S,mel_bank,P,logP,D
  
 
 
def mel_filter(M,N,fs,l,h):
 '''mel滤波器
 input:
   M：滤波器个数  这里取了24
   N：FFT点数
   fs：采样频率
   l(float)：低频系数  这里取了0
   h(float)：高频系数  这里取了0.5
 output:melbank(二维array):mel滤波器
 '''
 fl = fs * l #滤波器范围的最低频率
 fh = fs * h #滤波器范围的最高频率
 bl = 1125 * np.log(1 + fl / 700) #将频率转换为mel频率
 bh = 1125 * np.log(1 + fh /700) 
 B = bh - bl #频带宽度
 y = np.linspace(0,B,M+2) #将mel刻度等间距
 # print('mel间隔',y)
 Fb = 700 * (np.exp(y / 1125) - 1) #将mel变为HZ
 print(Fb)
 w2 = int(N / 2 + 1)
 df = fs / N
 freq = [] #采样频率值
 for n in range(0,w2):
  freqs = int(n * df)
  freq.append(freqs)
 melbank = np.zeros((M,w2))
 print(freq)
  
 for k in range(1,M+1):
  f1 = Fb[k - 1]
  f2 = Fb[k + 1]
  f0 = Fb[k]
  n1 = np.floor(f1/df)
  n2 = np.floor(f2/df)
  n0 = np.floor(f0/df)
  for i in range(1,w2):
   if i >= n1 and i <= n0:
    melbank[k-1,i] = (i-n1)/(n0-n1)
   if i >= n0 and i <= n2:
    melbank[k-1,i] = (n2-i)/(n2-n0)
  plt.plot(freq,melbank[k-1,:])
 # plt.show()
 plt.savefig('mel滤波器.png')
 return melbank,w2
 
if __name__ == '__main__':
 path = "E:\\MFCC_hw"
 name = 'shop11.wav'
 filename = os.path.join(path, name)
 data_path = filename
 #win = 256
 wlen = 256
 nfft = wlen 
 win = hanning_window(wlen)
 inc = 128
 wavedata,nframes,framerate,energy,zeroCrossingRate = read(data_path)
 FrameK = point_check(wavedata,win,inc,nframes,framerate,energy,zeroCrossingRate)
 S,mel_bank,P,logP,D = mfcc(FrameK,framerate,wlen)
 D = D.T
 #print("窗长：")
 #print(wlen)
 #print("MFCC的shape：")
 #print(D.shape[0])
 #print(D)
 '''
 time = np.arange(0,nframes) * (1.0 / framerate)
 time= np.reshape(time,[nframes,1]).T
 plt.plot(time[0,:nframes],wavedata[0,:nframes],c="b")
 plt.xlabel("time")
 plt.ylabel("amplitude")
 plt.title("shop")
 plt.show()

 with open("E:\\MFCC_hw\\"  + "mfcc.txt","w") as f :
        for i in range(D.shape[0]) :
            for j in range(D.shape[1]) :
                if j == 0 :
                    f.write("第"+str(i+1)+"个窗口的12位mfcc值：")
                f.write(str(D[i][j]) + " ")
            f.write("\n")
'''