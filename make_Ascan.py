import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft



def make_Ascan(i):
    # Read csv data
    if i == 0:
        data_name = 'Through.csv'
    else:
        data_name = 'TX5-RX' + str(i) + '.csv'
    data = pd.read_csv(data_name, header=None, skiprows=19)
    Time = data[0]  # Time [s]
    #Time = Time - min(Time) # -1~+1を0~+2に変換
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
    Amp_PSD = 10 * np.log10(Amp_ASD) # Amplitude Spectrum Density, [dB/Hz]
    Amp_norm = Amp_ASD / np.max(Amp_ASD[1:int(N/2)]) # normalize
    Amp_PSD_norm = 10 * np.log10(Amp_norm) # Power Spectrum Density normalized, [dB/Hz]



    # calculate tau
    freq_start = 0.3e9 # [Hz]
    freq_end = 1.2e9 # [Hz]
    sweep_rate = (freq_end - freq_start) / 1 # [Hz/s]
    tau = Freq / sweep_rate # delay time [s]


    # save data as csv
    data = np.vstack((tau, Amp, Amp_ASD, Amp_PSD_norm))
    data = data.T
    # add header
    header = ['tau [s]', 'AS [V]', 'ASD [V/√Hz]', 'PSD [dB/Hz]']
    data = np.vstack((header, data))

    out_dir = 'Ascan/' + data_name.split('.')[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(out_dir + '/tau_ASD_PSD.csv', data, delimiter=',', fmt='%s')

    """
    # plot
    plt.plot(tau[1:int(N/2)], Amp_PSD[1:int(N/2)])

    plt.title(data_name.split('.')[0], size = 16)
    plt.xlabel('Delay Time [s]', size = 14)
    plt.ylabel('PSD [dB/Hz]', size = 14)
    #plt.xscale('log')
    plt.xlim(0, 100e-9)
    plt.grid()

    plt.savefig(out_dir + '/tau_PSD.png', dpi=300)
    plt.show()
    """

    return tau, Amp, Amp_PSD, Amp_PSD_norm, out_dir

tau1, Amp1, Amp_PSD1, Amp_PSD_norm1, out_dir1 = make_Ascan(1)
tau2, Amp2, Amp_PSD2, Amp_PSD_norm2, out_dir2 = make_Ascan(2)
tau3, Amp3, Amp_PSD3, Amp_PSD_norm3, out_dir3 = make_Ascan(3)
tau4, Amp4, Amp_PSD4, Amp_PSD_norm4, out_dir4 = make_Ascan(4)
tau0, Amp0, Amp_PSD0, Amp_PSD_norm0, out_dir0 = make_Ascan(0)


N = len(tau1) # data size
# =====plot=====
plt.figure(figsize=(10, 10))
plt.plot(tau1[1:int(N/2)], Amp_PSD1[1: int(N/2)]-50, label='TX5-RX1')
plt.plot(tau2[1:int(N/2)], Amp_PSD2[1: int(N/2)]-100, label='TX5-RX2')
plt.plot(tau3[1:int(N/2)], Amp_PSD3[1: int(N/2)]-150, label='TX5-RX3')
plt.plot(tau4[1:int(N/2)], Amp_PSD4[1: int(N/2)]-200, label='TX5-RX4')
plt.plot(tau0[1:int(N/2)], Amp_PSD0[1: int(N/2)], label='Through')

plt.xlabel('Delay Time [s]', size = 14)
plt.ylabel('PSD [dB/Hz]', size = 14)
plt.xlim(0, 100e-9)
plt.ylim(-250, 50)
plt.grid()
plt.legend(fontsize=8)
plt.savefig('Ascan/Ascan_all', dpi=300)
plt.show()



def plot_1by1(tau, Amp_PSD, data_name, out_dir):
    plt.plot(tau[1:int(len(tau)/2)], Amp_PSD[1:int(len(Amp_PSD)/2)])

    plt.title(data_name.split('.')[0], size = 16)
    plt.xlabel('Delay Time [s]', size = 14)
    plt.ylabel('PSD [dB/Hz]', size = 14)
    #plt.xscale('log')
    plt.xlim(0, 100e-9)
    plt.grid()

    plt.savefig(out_dir + '/tau_PSD.png', dpi=300)
    plt.show()

    return plt
"""
plot_1by1(tau1, Amp_PSD1, 'TX5-RX1.csv', out_dir1)
plot_1by1(tau2, Amp_PSD2, 'TX5-RX2.csv', out_dir2)
plot_1by1(tau3, Amp_PSD3, 'TX5-RX3.csv', out_dir3)
plot_1by1(tau4, Amp_PSD4, 'TX5-RX4.csv', out_dir4)
plot_1by1(tau0, Amp_PSD0, 'Through.csv', out_dir0)
"""