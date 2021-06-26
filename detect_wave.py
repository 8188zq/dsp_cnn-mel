import wave
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

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

# 计算每一帧的能量 
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
    print(x1)
    print(x2)
    for i in range(x1) :
        for j in range(x2) :
            if j == 0 :
                continue
            sum = sum + np.abs(sgn(fdata[i][j]) - sgn(fdata[i][j - 1]))/2
        zeroCrossingRate.append(float(sum) / x2)
        sum = 0
    return zeroCrossingRate

# 利用短时能量，短时过零率，使用双门限法进行端点检测
def endPointDetect(wave_data, energy, zeroCrossingRate) :
    sum = 0
    energyAverage = 0
    for en in energy :
        sum = sum + en
    energyAverage = sum / len(energy)
  ### 注意下面的参数都是调整的，效果最好的一组
    sum = 0
    for en in energy[:5] :
        sum = sum + en
    ML = sum / 5                        
    MH = energyAverage / 4             #较高的能量阈值
    ML = (ML + MH) / 4    #较低的能量阈值    
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
    print("较高能量阈值，计算后的浊音A:" + str(A))

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
    print("较低能量阈值，增加一段语言B:" + str(B))

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
    print("过零率阈值，最终语音分段C:" + str(C))
    return C


path = "E:\\MFCC_hw"
name = 'shop4.wav'
filename = os.path.join(path, name)
f = wave.open(filename,'rb')
params = f.getparams()
nchannels, sampwidth, framerate,nframes = params[:4]
strData = f.readframes(nframes)
waveData = np.fromstring(strData,dtype=np.short)
# waveData = waveData * 1.0/max(abs(waveData))
# waveData = np.reshape(waveData,[nframes,nchannels]).T
f.close()
print("wavadata之前的shape:")
print(waveData.shape)
time = np.arange(0,nframes) * (1.0 / framerate)
time= np.reshape(time,[nframes,1]).T


wlen = 256
nfft = wlen
win = hanning_window(wlen)
inc = 128
energy = calEnergy(waveData,win,inc)
zeroCrossingRate = calZeroCrossingRate(waveData,win,inc)
N = endPointDetect(waveData, energy, zeroCrossingRate)
i = 0
'''
while i < len(N) :
    for num in waveData[N[i] * 256 : N[i+1] * 256] :
        f.write(num)
    i = i + 2
'''

waveData = waveData * 1.0/max(abs(waveData))
waveData = np.reshape(waveData,[nframes,nchannels]).T
print("wavedata之后的shape:")
print(waveData.shape)
plt.plot(time[0,:nframes],waveData[0,:nframes],c="b")
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("Original wave_two")
plt.vlines(N[0]*inc* (1.0 / framerate), -1.0, 1.0, colors = "c", linestyles = "dashed")
plt.vlines(N[len(N)-1]*inc* (1.0 / framerate), -1.0, 1.0, colors = "c", linestyles = "dashed")
plt.show()