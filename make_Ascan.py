import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft

# ======load files=====
# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate delay time and plot A-scan.', 
                                 usage='cd bs-radar; python make_Ascan.py plot_type')
parser.add_argument('plot_type', choices=['3panels', '4panels', 'single'], help='plot type')
args = parser.parse_args()



for i in range(5):
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
    tau = Freq / sweep_rate # delay time not consider delay in cable [s]

    cable_delay = 4.448784722222222e-08 # delay time while signal travels through cable [s]
    tau_travel = tau - cable_delay # delay time consider delay in cable [s]

    tau_travel_0index = np.where(tau_travel >= 0)[0][0] # index where tau_travel get 0 for the first time
    print('tau_travel_0index =', tau_travel_0index)
    tau_travel = tau_travel[tau_travel_0index:int(N/2)] # cut off tau_travel before 0
    Amp_travel = Amp[tau_travel_0index:int(N/2)] # cut off Amp before 0
    Amp_ASD_travel = Amp_ASD[tau_travel_0index:int(N/2)] # cut off Amp_ASD before 0
    Amp_PSD_travel = 10 * np.log10(Amp_ASD_travel) # Amplitude Spectrum Density, [dB/Hz]
    Amp_norm_travel = Amp_ASD_travel / np.max(Amp_ASD_travel) # normalize
    Amp_PSD_norm_travel = 10 * np.log10(Amp_norm_travel) # cut off Amp_PSD_norm before 0


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


    # =====plot=====
    def plot_3panel():
        fig, ax = plt.subplots(3, 1, sharex='all', tight_layout=True, figsize=(10, 10))

        ax[0].plot(tau[1:int(N/2)], Amp[1:int(N/2)])
        ax[0].set_ylabel('Amplitude [V]', size = 14)
        ax[0].grid()

        ax[1].plot(tau[1:int(N/2)], Amp_PSD[1:int(N/2)])
        ax[1].set_ylabel('Power [dB]', size = 14)
        ax[1].grid()

        ax[2].plot(tau[1:int(N/2)], Amp_PSD_norm[1:int(N/2)])
        ax[2].set_ylabel('Normalized Power [dB]', size = 14)
        ax[2].grid()

        fig.suptitle(data_name.split('.')[0], size = 16)
        fig.supxlabel('Delay Time [s]', size = 14)
        plt.xlim(0, 100e-9)

        plt.savefig(out_dir + '/tau_ASD_PSD.png', dpi=300)
        plt.show()

        return plt


    def plot_4panel():
        fig, ax = plt.subplots(4, 1, sharex='all', tight_layout=True, figsize=(10, 10))

        ax[0].plot(tau[1:int(N/2)], Amp[1:int(N/2)])
        ax[0].set_ylabel('Amplitude [V]', size = 14)
        ax[0].grid()

        ax[1].plot(tau[1:int(N/2)], Amp_PSD_norm[1:int(N/2)])
        ax[1].set_ylabel('Normalized Power [dB]', size = 14)
        ax[1].grid()

        ax[2].plot(tau_travel, Amp_travel)
        ax[2].set_ylabel('Amplitude [V]', size = 14)
        ax[2].grid()

        ax[3].plot(tau_travel, Amp_PSD_norm_travel)
        ax[3].set_ylabel('Normalized Power [dB]', size = 14)
        ax[3].grid()

        fig.suptitle(data_name.split('.')[0], size = 16)
        fig.supxlabel('Delay Time [s]', size = 14)
        plt.xlim(0, 100e-9)

        plt.savefig(out_dir + '/compare_tau.png', dpi=300)
        plt.show()

        return plt



    def plt_PSD():
        fig, ax = plt.subplots(5, 1, tight_layout=True, figsize=(10, 10), sharex='all')
        ax[i].set_title(data_name.split('.')[0], size = 16)
        ax[i].plot(tau[1:int(N/2)], Amp_PSD_norm[1:int(N/2)])
        ax[i].grid()
        ax[i].set_xlim(0, 100e-9)

        fig.supxlabel('Delay Time [s]', size = 14)
        fig.supylabel('Normalized Power [dB]', size = 14)


        plt.savefig(out_dir + '/tau_PSD.png', dpi=300)
        plt.show()

        return plt
    
    if args.plot_type == '3panels':
        plot_3panel()
    elif args.plot_type == '4panels':
        plot_4panel()
    elif args.plot_type == 'single':
        plt_PSD()


"""
def Ascan_plot_1sheet():
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

    return plt
"""




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