import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft

# plot数の上限を上げるおまじない（？）
plt.rcParams['agg.path.chunksize'] = 10000


# Read csv data
data_name = 'TX5-RX1.csv'
data = pd.read_csv(data_name, header=None, skiprows=19)
Time = data[0] + 1 # Time [s]
Input = data[1] # Voltage applied to VCO [V]
Output = np.array(data[2]) # IF output [V], np.array()にしないとFFTでエラーが出る（なんで？）


# check index where Input get 2.00 for the first time
index = np.where(Input >= 2.00)[0][0]
t_loc = np.where(Input >= 2.00)[0][0] * (1 / len(Input)) # tau_lac [s]



# =====FFT=====
# set parameters
fs = 16000 # sampling frequency [Hz]
N = len(Output) # data size

# calculate FFT
out_fft = fft.fft(Output)
Amp = np.abs(out_fft)
Freq = fft.fftfreq(N, 1/fs)

# calculate phase
phi_f = np.angle(out_fft)
phi_f = np.where(phi_f < 0, phi_f + 2 * np.pi, phi_f)

# calculate time
sweep_rate = 0.9e9 / 1 # [Hz/s]
tau_f = np.sqrt(phi_f / (2 * np.pi * sweep_rate)) # [s],　← 怪しいことしてる



# =====IFFT=====
tau_t = fft.ifft(tau_f)
tau_t = np.real(tau_t)
tau_t = tau_t / Time # [s], ← 怪しいことしてる
#plt.plot(Time, tau_t, 'o')
#plt.plot(tau_t, Output)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(0, 2)
#plt.show()


# make tau_t vs Output plot
tau_t_sort = tau_t[np.argsort(tau_t)]
Output_sort = Output[np.argsort(tau_t)]


# =====save data and plot=====
# save data
save_dir_path = 'Ascan/' + data_name.split('.')[0] + '.csv'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
data_save = np.vstack((tau_t_sort, Output_sort))
data_save = data_save.T
np.savetxt(save_dir_path + '/tau_t_Output.csv', data_save, delimiter=',')

# plot
plt.plot(tau_t_sort, Output_sort)

plt.title(data_name.split('.')[0], size = 16)
plt.xlabel('Time [s]', size = 14)
plt.ylabel('Output Voltage [V]', size = 14)
plt.ylim(0.18, 0.30)
plt.xscale('log')
#plt.xlim(0, 1)
plt.grid()
plt.savefig(save_dir_path + '/tau_t_Output.png', dpi=300)
plt.show()


print('saved')



# 以下，余計かもしれない
# make t-phase data
phi_t = fft.ifft(phi_f)
phi_t = np.real(phi_t)
phi_t = np.where(phi_f < 0, phi_f + 2 * np.pi, phi_f)
#plt.plot(Time, phi_t, 'o')
#plt.show()

# calculate delay time

#tau = phi_t / (2 * np.pi * sweep_rate) # [s]

# assume tau^2 << 1
#tau = phi_t / (2 * np.pi) \
#    / (sweep_rate * Time - sweep_rate * t_lac + 0.3e9) # [s]

# consider tqau^2
term1 = phi_t / sweep_rate / np.pi
tau =  term1 + np.sqrt( \
    term1**2 + Time - t_loc + 0.3e9 / sweep_rate)

#plt.plot(phi_t, Output)
#plt.plot(tau, phi_t)
#plt.plot(Time, tau, 'o')
#plt.xscale('log')
#plt.show()







#plt.plot(Freq[1:int(N/2)], np.abs(Amp[1:int(N/2)]))
def plt_Freq_data():
    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(8, 6), sharex='none')
    ax[0].plot(Freq[1:int(N/2)], Amp[1:int(N/2)])
    ax[0].set_xlabel('Frequency [Hz]', size = 14)
    ax[0].set_ylabel('Amplitude', size = 14)
    ax[0].set_xscale('log')

    ax[1].plot(Freq[1:int(N/2)], phi_f[1:int(N/2)])
    ax[1].set_xlabel('Frequency [Hz]', size = 14)
    ax[1].set_ylabel('Phase', size = 14)
    ax[1].set_xscale('log')

    ax[2].plot(Freq[1:int(N/2)], phi_t[1:int(N/2)])
    ax[2].set_xlabel('Frequency [Hz]', size = 14)
    ax[2].set_ylabel('Time [s]', size = 14)
    ax[2].set_xscale('log')

    plt.show()
#plt_Freq_data()


def plt_time_data():
    fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(8, 6), sharex='none')
    ax[0].plot(Time, Output)
    ax[0].set_ylabel('Amplitude', size = 14)

    ax[1].plot(Time, phi_t)
    ax[1].set_ylabel('Phase', size = 14)
    ax[1].set_xscale('log')

    plt.xlabel('Time [s]')
    plt.show()
#plt_time_data()