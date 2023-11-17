import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft

# Read csv data
data_51 = pd.read_csv('TX5-RX1.csv', header=None, skiprows=19)
Time = np.array(data_51[0]) # Time [s]
out_51 = np.array(data_51[2]) # IF output [V]

# check spectrum
fs = 16000 # sampling frequency [Hz]
N = len(out_51) # data size
t = np.arange(0, N/fs, 1/fs)
out_fft = fft.fft(out_51)
Amp = np.abs(out_fft/(N/2))
Freq = fft.fftfreq(N, 1/fs)

sweep_rate = 0.9e9 / 2 # [Hz/s]
delay_time = out_fft / sweep_rate

# IFFT
out_ifft = fft.ifft(delay_time)
out_ifft = np.real(out_ifft)


#plt.plot(Freq[1:int(N/2)], np.abs(delay_time[1:int(N/2)]))
plt.plot(Time, out_ifft)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
#plt.xscale('log')
plt.show()