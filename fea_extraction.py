import numpy as np
import librosa
import librosa.display
import os
import warnings
warnings.filterwarnings("error")


padto = 50000

### 调用librosa库，非常好用
def mel_spec(path):

    y, sr = librosa.load(path)
    # y = y * 1.0 / (max(abs(y)))
    # print(y.shape)
    # print(type(y))
    # print(max(y))
    l = len(y)
    if l<=padto:
        left_pad = (padto - l) // 2
        righ_pad = padto - l - left_pad
        y = np.pad(y, (left_pad, righ_pad), 'wrap')
    else:
        left = (l-padto)//2
        right = left+padto 
        y = y[left:right]
    

    # librosa.display.waveplot(y, sr=sr)

    # pre_shape = y.shape

    feat = librosa.stft(y, hop_length=512, n_fft=1024)
    feat = np.abs(feat) ** 2
    try:
        feat = librosa.feature.melspectrogram(S=feat, sr=sr, n_mels=128)
    except:
        raise IOError(path)
    #     import ipdb
    #     ipdb.set_tracSe()
    feat = librosa.power_to_db(feat, ref=np.max)
    # print(pre_shape, feat.shape)
    return feat

def specshow(feat):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feat,
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()

