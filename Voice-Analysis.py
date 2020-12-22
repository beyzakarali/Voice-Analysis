#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf



SAMPLING_RATE = 16000
data =   []
classes = 4  # 4 human voice
noise_numb = 2
current_path = r"C:\Users\Beyza\Desktop\16000_pcm_speeches"


def load_noise_sample(path):
    
    sample , sampling_rate = librosa.load(path , sr=16000) 
    return sample


for i in range (noise_numb):
    path = os.path.join(current_path + '\\noise\\' + str(i)) 
    noisepath = os.listdir(path)  
    print(i,".dosya path->",noisepath)
    noises = []
    
    
    for path_new in noisepath:
       
            sample = load_noise_sample(path +"\\" +path_new)
            print(path_new,". file load:" ,sample)
            print("Spectogram Func")
            spectrogram(sample,sampling_rate)
            print("Specgram func")
            specgram(sample , sampling_rate)
            

            


           


            
        


# Bu kütüphaneyle işleyeceğimiz sesi dinleyebiliriz.

# In[4]:


from IPython.display import Audio
Audio(r"C:\Users\Beyza\Desktop\0.wav")


# Sesi görselleştirmemizi sağladı.

# In[5]:


from librosa import display
sample , sampling_rate = librosa.load(r"C:\Users\Beyza\Desktop\0.wav" , sr=16000)

plt.figure()
librosa.display.waveplot(y=sample,sr=sampling_rate)
plt.xlabel("Time (seconds)-->")
plt.ylabel("Amplitude")
plt.show()


# Bir sinüs dalgası üzerinde FFT uygulama

# In[6]:


import numpy as np

# Sampling rate =100, frekans=11, genlik/amplitude =2
samples = 100
f = 11
x = np.arange(samples)
y2 = 2 * np.sin(2*np.pi*f* (x/samples))
plt.figure()
plt.stem(x,y2, 'r', ) # for points
plt.plot(x,y2)
plt.xlabel("Time-->")
plt.ylabel("<----amplitude--->")
plt.show()


# In[65]:


import scipy
def fft_plot(audio , sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft(audio)
    xf = np.linspace(0.0 , 1.0/(2.0*T) , n/2.0)
    fig, ax = plt.subplots()
    ax.plot(xf , 2.0/n* np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency -->")
    plt.ylabel("Magnitude")
    return plt.show()

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
sample , sampling_rate = librosa.load(r"C:\Users\Beyza\Desktop\0.wav" , sr=16000)    
#Another spectogram func
plt.specgram(sample, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')   


# Spectogram fonksiyonu bir ses için vektör döndürüyor.

# In[77]:


def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = 20, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size                #Kesim sayısı
    samples = samples[:len(samples) - truncate_size]                          #Windows sayısı
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1) 
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    print(specgram)
    return specgram


# In[67]:


# Gösterim için yukardaki fonksiyon kullanılmadı.
def specgram(sample , sampling_rate):
   # sample , sampling_rate = librosa.load(r"C:\Users\Beyza\Desktop\0.wav", sr=16000) 
   #%matplotlib inline
    X = librosa.stft(sample)                     #fft uygulama 
    print(X)
    Y = spectrogram(sample,sampling_rate)       #ignoring
    print(len(Y))
    Xdb = librosa.amplitude_to_db(abs(X))       # = power_to_db(S**2) -> scaling sağlıyor

    plt.figure(figsize=(14, 5))                 #görselleştirme
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.colorbar()


# In[35]:


n_simple1 = spectrogram(sample,sampling_rate) #FFT sonucu karmaşık sayılar listesi oluşur.
print(len(n_simple1)) #fft nin çıkışı
#fft_plot(n_simple1 , sampling_rate) 


# In[55]:


#n_sample = spectrogram(sample , sampling_rate)
sample , sampling_rate = librosa.load(r"C:\Users\Beyza\Desktop\0.wav" , sr=16000)
print(sample)

n_sample = librosa.stft(sample)                  # serial fft uygulandı.#db_ tarzı spectogram deniliyor
print("n_sample",n_sample)
librosa.power_to_db(n_sample**2, ref=np.median)  #db_ den power spectogramına çevirme

plt.figure(figsize=(10, 4))                     #görselleştirme mevzuları
librosa.display.specshow(n_sample)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

