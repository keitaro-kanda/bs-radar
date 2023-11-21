# FFT exercise
from re import A

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft

# set parameters
fs = 16000 # sampling frequency [Hz]
N = 16000 # data size
t = np.arange(0, N/fs, 1/fs) # time [s]

# make signal
f1 = 10 # [Hz]
f2 = 20 # [Hz]
f3 = 50 # [Hz]
f4 = 75 # [Hz]
f5 = 100 # [Hz]

A1 = 1
A2 = 0.5
A3 = 1
A4 = 1.2
A5 = 0.1

signal = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) \
    + A3 * np.sin(2 * np.pi * f3 * t) + A4 * np.sin(2 * np.pi * f4 * t) \
    + A5 * np.sin(2 * np.pi * f5 * t)

# calculate FFT
signal_fft = fft.fft(signal) / (N/2)
Amp = np.abs(signal_fft)
Freq = fft.fftfreq(N, 1/fs)

# plot
fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6))
ax[0].plot(t, signal)
ax[0].set_xlabel('Time [s]', size = 14)
ax[0].set_ylabel('Amplitude', size = 14)

ax[1].plot(Freq[1:int(N/2)], Amp[1:int(N/2)])
ax[1].set_xlabel('Frequency [Hz]', size = 14)
ax[1].set_ylabel('Amplitude', size = 14)
ax[1].set_xscale('log')

plt.show()