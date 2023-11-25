import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft

# Read csv data
for i in range(1, 6):
    if i == 5:
        data_name = 'Through.csv'
    else:
        data_name = 'TX5-RX' + str(i) + '.csv'
    data = pd.read_csv(data_name, header=None, skiprows=19)
    Time = data[0]  # Time [s]
    Time = Time - min(Time) # -1~+1を0~+2に変換
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
    Freq = fft.fftfreq(N, 1/fs)
    Amp = np.abs(out_fft) # Amplitude Spectrum, [V]
    Amp_ASD = np.sqrt(Amp**2 / (fs/N)) # Amplitude Spectrum Density, [V/√Hz]
    Amp_norm = Amp_ASD / np.max(Amp_ASD[1:int(N/2)]) # normalize
    Amp_PSD = 10 * np.log10(Amp_norm) # Power Spectrum Densitt, [dB/Hz]



    # calculate tau
    freq_start = 0.3e9 # [Hz]
    freq_end = 1.2e9 # [Hz]
    sweep_rate = (freq_end - freq_start) / 1 # [Hz/s]
    tau = Freq / sweep_rate # delay time [s]


    # save data as csv
    data = np.vstack((tau, Amp, Amp_ASD, Amp_PSD))
    data = data.T
    # add header
    header = ['tau [s]', 'AS [V]', 'ASD [V/√Hz]', 'PSD [dB/Hz]']
    data = np.vstack((header, data))

    out_dir = 'Ascan/' + data_name.split('.')[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(out_dir + '/tau_ASD_PSD.csv', data, delimiter=',', fmt='%s')

    # plot
    plt.plot(tau[1:int(N/2)], Amp_PSD[1:int(N/2)])

    plt.title(data_name.split('.')[0], size = 16)
    plt.xlabel('Delay Time [s]', size = 14)
    plt.ylabel('PSD [dB/Hz]', size = 14)
    plt.xscale('log')
    plt.grid()

    plt.savefig(out_dir + '/tau_PSD.png', dpi=300)
    plt.show()