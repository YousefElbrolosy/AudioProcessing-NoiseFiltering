
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import sounddevice as sd
leftHand = [0,0,0,0,0,0]
rightHand = [261.63,293.66,311.13,349.23,392,466.16]

t = np.linspace(0 ,12 , 16 * 1024)

i = 0
ti = 0
tp = 0.75
sum = 0


while(i<len(rightHand)):
    
    z1 = np.reshape(np.sin(2*np.pi*rightHand[i]*t)*([t>=ti]),np.shape(t))
    z2 = np.reshape(np.sin(2*np.pi*rightHand[i]*t)*([t>=ti+tp]),np.shape(t))
    z = z1-z2

    sum+=z
    x1 = np.reshape(np.sin(2*np.pi*leftHand[i]*t)*([t>=ti]),np.shape(t))
    x2 = np.reshape(np.sin(2*np.pi*leftHand[i]*t)*([t>=ti+tp]),np.shape(t))
    x = x1-x2

    sum+=x
    i+=1
    ti+=0.75
    

ti+=0.2
tp = 1

    
z1 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti+tp]),np.shape(t))
z = z1-z2

sum+=z

z1 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti+tp]),np.shape(t))
z = z1-z2

sum+=z

ti+=1

z1 = np.reshape(np.sin(2*np.pi*392*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*392*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

ti+=1

z1 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

z1 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

ti+=0.75

z1 = np.reshape(np.sin(2*np.pi*392*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*392*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

ti+=0.75


z1 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti+0.37]),np.shape(t))
z = z1-z2

sum+=z

z1 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti+0.37]),np.shape(t))
z = z1-z2

sum+=z

ti+=0.6


z1 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*349.23*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

z1 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti]),np.shape(t))
z2 = np.reshape(np.sin(2*np.pi*440*t)*([t>=ti+0.5]),np.shape(t))
z = z1-z2

sum+=z

f1 = np.random.randint(0, 512)
f2 = np.random.randint(0, 512)
N=16*1024
f=np.linspace(0,512,int(N/2))

noise1 = np.sin(2 * f1 * np.pi * t)

noise2 = np.sin(2 * f2 * np.pi * t)

noise = noise1+noise2
# this plots sum
fig,ax=plt.subplots(2,3)

ax[0, 0].plot(t,sum)

ax[0,0].set_title ('T Domain without noise')

#///////////////

# this plots F domain

freq_data= fft(sum)
x_f= 2/N *np.abs(freq_data [0:int(N/2)])

ax[1,0].plot(f,x_f)

ax[1,0].set_title ('F Domain without noise')

#////////////////

# this plots T with noise



ax[0,1].plot(t, sum+noise)

ax[0,1].set_title('T Domain with noise')

#///////////////////

# this plots F with noise

freq_data_noise = fft(sum + noise)
x_f_noise = 2 / N * np.abs(freq_data_noise[0:int(N / 2)])

ax[1,1].plot(f, x_f_noise)

ax[1,1].set_title ('F Domain with noise')

#/////////////////////////////
# this plots Time domain filtered

temp=np.sort(x_f_noise)
freq1=round(temp[-1])
freq2=round(temp[-2])



# &awel freq dom, sort then 7awel back l time

noise1new=np.sin(2 * freq1 * np.pi * t)
noise2new=np.sin(2 * freq2 * np.pi * t)


noisenew=noise1new+noise2new

freqNewNoise = fft(noisenew)
x_freqNewNoise = 2 / N * np.abs(freqNewNoise[0:int(N)])
temp = np.sort(x_freqNewNoise)
noiseNewSorted = np.abs(ifft(temp))

freqNoise = fft(noise)
x_freqNoise = 2 / N * np.abs(freqNoise[0:int(N)])
temp1 = np.sort(x_freqNoise)
noiseSorted = np.abs(ifft(temp1))

xfiltered= sum + noiseSorted - noiseNewSorted


temp=np.sort(x_f_noise)






ax[0,2].plot(t, xfiltered)

ax[0,2].set_title ('T Domain filtered')

freq_data_filtered = fft(xfiltered)
x_f_filtered = 2 / N * np.abs(freq_data_filtered[0:int(N / 2)])



ax[1,2].plot(f,x_f_filtered)

ax[1, 2].set_title ('F Domain filtered')




fig.tight_layout(pad=2)
sd.play(xfiltered,2*1024)
plt.show()