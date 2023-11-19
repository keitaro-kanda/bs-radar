import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft
plt.rcParams['agg.path.chunksize'] = 10000

# Read csv data
data_51 = pd.read_csv('TX5-RX1.csv', header=None, skiprows=19)
Time = data_51[0] + 1 # Time [s]
Input = data_51[1] # Voltage applied to VCO [V]
# check index where Input get 2.00 for the first time
index = np.where(Input >= 2.00)[0][0]
t_loc = np.where(Input >= 2.00)[0][0] * (1 / len(Input)) # tau_lac [s]
out_51 = np.array(data_51[2]) # IF output [V]

# FFT
fs = 16000 # sampling frequency [Hz]
N = len(out_51) # data size
#t = np.arange(0, N/fs, 1/fs)

out_fft = fft.fft(out_51)
Amp = np.abs(out_fft)
Freq = fft.fftfreq(N, 1/fs)

# calculate phase and check the time
f_phase = np.angle(out_fft)
f_phase = np.where(f_phase < 0, f_phase + 2 * np.pi, f_phase)

# calculate time
sweep_rate = 0.9e9 / 1 # [Hz/s]
#tau = np.sqrt(f_phase / (2 * np.pi * sweep_rate)) # [s]
#plt.plot(tau, f_phase)
#plt.show()


# make t-phase data
t_phase = fft.ifft(f_phase)
t_phase = np.real(t_phase)
t_phase = np.where(f_phase < 0, f_phase + 2 * np.pi, f_phase)
#plt.plot(Time, t_phase, 'o')
#plt.show()

# calculate delay time

#tau = t_phase / (2 * np.pi * sweep_rate) # [s]

# assume tau^2 << 1
#tau = t_phase / (2 * np.pi) \
#    / (sweep_rate * Time - sweep_rate * t_lac + 0.3e9) # [s]

# consider tqau^2
term1 = t_phase / sweep_rate / np.pi
tau =  term1 + np.sqrt( \
    term1**2 + Time - t_loc + 0.3e9 / sweep_rate)

#plt.plot(t_phase, out_51)
#plt.plot(tau, t_phase)
plt.plot(Time, tau, 'o')
#plt.xscale('log')
plt.show()







#plt.plot(Freq[1:int(N/2)], np.abs(Amp[1:int(N/2)]))
def plt_Freq_data():
    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(8, 6), sharex='none')
    ax[0].plot(Freq[1:int(N/2)], Amp[1:int(N/2)])
    ax[0].set_xlabel('Frequency [Hz]', size = 14)
    ax[0].set_ylabel('Amplitude', size = 14)
    ax[0].set_xscale('log')

    ax[1].plot(Freq[1:int(N/2)], f_phase[1:int(N/2)])
    ax[1].set_xlabel('Frequency [Hz]', size = 14)
    ax[1].set_ylabel('Phase', size = 14)
    ax[1].set_xscale('log')

    ax[2].plot(Freq[1:int(N/2)], t_phase[1:int(N/2)])
    ax[2].set_xlabel('Frequency [Hz]', size = 14)
    ax[2].set_ylabel('Time [s]', size = 14)
    ax[2].set_xscale('log')

    plt.show()
#plt_Freq_data()


def plt_time_data():
    fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6), sharex='none')
    ax[0].plot(Time, out_51)
    ax[0].set_ylabel('Amplitude', size = 14)

    ax[1].plot(Time, t_phase)
    ax[1].set_ylabel('Phase', size = 14)
    ax[1].set_xscale('log')

    plt.xlabel('Time [s]')
    plt.show()
#plt_time_data()