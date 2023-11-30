import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft

# ======load files=====
# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate delay time and plot A-scan.', 
                                usage='cd bs-radar; python -m tools.make_Ascan plot_type')
parser.add_argument('plot_type', choices=['raw', 'travel'], help='plot type')
args = parser.parse_args()



class make_Ascan:
    def __init__(self):
        self.data_name = ''
        #self.data = None
        #self.Time = None
        #self.Input = None
        #self.Output = None
    
    def calc_tau(self):
        data_dir = 'source_data/'
        self.data = pd.read_csv(data_dir + self.data_name, header=None, skiprows=19)
        self.Time = self.data[0]
        self.Input = self.data[1]
        self.Output = self.data[2]


        # =====FFT=====
        # set parameters
        fs = 16000
        N = len(self.Output)

        # calculate FFT
        out_fft = fft.fft(self.Output.values) # .valuesをつけるとSeries型からndarray型に変換できる
        self.Freq = fft.fftfreq(N, 1/fs) # [Hz]
        self.Amp = np.abs(out_fft) # Amplitude Spectrum, [V]
        self.ASD = np.sqrt(self.Amp**2 / (fs/N)) # Amplitude Spectrum Density, [V/√Hz]
        self.PSD = 10 * np.log10(self.ASD) # power Spectrum Density, [dB/Hz]
        self.PSD_norm = self.ASD / np.max(self.ASD[1:int(N/2)]) # normalize
        self.PSD_norm = 10 * np.log10(self.PSD_norm) # Power Spectrum Density normalized, [dB/Hz]

        # running average
        self.Amp_ave = pd.Series(self.Amp).rolling(2, min_periods=1).mean().values

        # =====calculate tau=====
        freq_start = 0.3e9
        freq_end = 1.2e9
        sweep_rate = (freq_end - freq_start) / 1
        self.tau = self.Freq / sweep_rate # delay time not consider delay in cable [s]

        # consider delay in cable
        cable_delay = 4.448784722222222e-08 # delay time while signal travels through cable [s]
        #cable_delay2 = 4.503038194444444e-08 # delay time while signal travels through cable [s]
        #cable_delay = (cable_delay1 + cable_delay2) / 2
        self.tau_travel = self.tau - cable_delay
        self.tau_travel_0index = np.where(self.tau_travel >= 0)[0][0]

        self.tau_travel = self.tau_travel[self.tau_travel_0index:int(N/2)] # cut off tau_travel before 0
        self.Amp_travel = self.Amp[self.tau_travel_0index:int(N/2)] # Amplitude Spectrum [V]
        self.ASD_travel = self.ASD[self.tau_travel_0index:int(N/2)] # Amplitude Spectrum Density, [V/√Hz]
        self.PSD_travel = 10 * np.log10(self.ASD_travel) # power Spectrum Density, [dB/Hz]
        self.PSD_norm_travel = self.ASD_travel / np.max(self.ASD_travel) # normalize
        self.PSD_norm_travel = 10 * np.log10(self.PSD_norm_travel) # Power Spectrum Density normalized, [dB/Hz]

        # running average
        self.Amp_travel_ave = pd.Series(self.Amp_travel).rolling(2, min_periods=1).mean().values
        

RX1 = make_Ascan()
RX1.data_name = 'TX5-RX1.csv'
RX1.calc_tau()

RX2 = make_Ascan()
RX2.data_name = 'TX5-RX2.csv'
RX2.calc_tau()

RX3 = make_Ascan()
RX3.data_name = 'TX5-RX3.csv'
RX3.calc_tau()

RX4 = make_Ascan()
RX4.data_name = 'TX5-RX4.csv'
RX4.calc_tau()

Through = make_Ascan()
Through.data_name = 'Through.csv'
Through.calc_tau()


# =====save data as csv=====
def save_data(RX):
    data = np.vstack((RX.tau_travel, RX.Amp_travel, RX.Amp_travel_ave, RX.PSD_norm_travel))
    data = data.T
    # add header
    header = ['2way travel time [s]', 'AS [V]', 'Running average of AS [V]',  'PSD [dB/Hz]']
    data = np.vstack((header, data))

    out_dir = 'results/Ascan/' + RX.data_name.split('.')[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(out_dir + '/Ascan_data_travel.csv', data, delimiter=',', fmt='%s')

save_data(RX1)
save_data(RX2)
save_data(RX3)
save_data(RX4)
save_data(Through)

# =====plot=====
def plot_Ascan():
    fig, ax = plt.subplots(5, 1, sharex='all', sharey='all', tight_layout=True, figsize=(10, 10))

    RXs = [RX1, RX2, RX3, RX4, Through]
    titles = ['TX5-RX1', 'TX5-RX2', 'TX5-RX3', 'TX5-RX4', 'Through']

    for i, (RX, title) in enumerate(zip(RXs, titles)):
        ax[i].plot(RX.tau[1:int(len(RX.tau)/2)], RX.Amp[1:int(len(RX.Amp)/2)])
        ax[i].plot(RX.tau[1:int(len(RX.tau)/2)], RX.Amp_ave[1:int(len(RX.Amp_ave)/2)])
        ax[i].set_title(title, size=16)
        ax[i].grid()

    #fig.supxlabel('Delay Time [s]', size = 14)
    #fig.supylabel('Amplitude [V]', size = 14)
    plt.xlim(0, 100e-9)
    plt.ylim(0, 45)
    #plt.xscale('log')

    plt.savefig('results//Ascan_raw.png', dpi=300)
    plt.show()
    return plt


def plot_Ascan_travel():
    fig, ax = plt.subplots(5, 1, sharex='all', sharey='all', tight_layout=True, figsize=(10, 10))

    RXs = [RX1, RX2, RX3, RX4, Through]
    titles = ['TX5-RX1', 'TX5-RX2', 'TX5-RX3', 'TX5-RX4', 'Through']

    for i, (RX, title) in enumerate(zip(RXs, titles)):
        ax[i].plot(RX.tau_travel, RX.Amp_travel)
        ax[i].plot(RX.tau_travel, RX.Amp_travel_ave)
        ax[i].set_title(title, size=16)
        ax[i].grid()

    fig.suptitle('A-scan travel', size = 16)
    #fig.supxlabel('Delay Time [s]', size = 14)
    #fig.supylabel('Amplitude [V]', size = 14)
    plt.xlim(0, 20e-9)
    plt.ylim(0, 45)
    #plt.xscale('log')

    plt.savefig('results/Ascan/Ascan_travel.png', dpi=300)
    plt.show()
    return plt

if args.plot_type == 'raw':
    plot_Ascan()
elif args.plot_type == 'travel':
    plot_Ascan_travel()
else:
    print('Error: plot_type is invalid. Choose raw or travel.')

"""
# 以下，元のコード
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
    ASD = np.sqrt(Amp**2 / (fs/N)) # Amplitude Spectrum Density, [V/√Hz]
    PSD = 10 * np.log10(ASD) # Amplitude Spectrum Density, [dB/Hz]
    PSD_norm = ASD / np.max(ASD[1:int(N/2)]) # normalize
    PSD_norm = 10 * np.log10(PSD_norm) # Power Spectrum Density normalized, [dB/Hz]



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
    ASD_travel = ASD[tau_travel_0index:int(N/2)] # cut off ASD before 0
    PSD_travel = 10 * np.log10(ASD_travel) # Amplitude Spectrum Density, [dB/Hz]
    PSD_norm_travel = ASD_travel / np.max(ASD_travel) # normalize
    PSD_norm_travel = 10 * np.log10(PSD_norm_travel) # cut off PSD_norm before 0


    # save data as csv
    data = np.vstack((tau, Amp, ASD, PSD_norm))
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

        ax[1].plot(tau[1:int(N/2)], PSD[1:int(N/2)])
        ax[1].set_ylabel('Power [dB]', size = 14)
        ax[1].grid()

        ax[2].plot(tau[1:int(N/2)], PSD_norm[1:int(N/2)])
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

        ax[1].plot(tau[1:int(N/2)], PSD_norm[1:int(N/2)])
        ax[1].set_ylabel('Normalized Power [dB]', size = 14)
        ax[1].grid()

        ax[2].plot(tau_travel, Amp_travel)
        ax[2].set_ylabel('Amplitude [V]', size = 14)
        ax[2].grid()

        ax[3].plot(tau_travel, PSD_norm_travel)
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
        ax[i].plot(tau[1:int(N/2)], PSD_norm[1:int(N/2)])
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

"""
def Ascan_plot_1sheet():
    N = len(tau1) # data size
    # =====plot=====
    plt.figure(figsize=(10, 10))
    plt.plot(tau1[1:int(N/2)], PSD1[1: int(N/2)]-50, label='TX5-RX1')
    plt.plot(tau2[1:int(N/2)], PSD2[1: int(N/2)]-100, label='TX5-RX2')
    plt.plot(tau3[1:int(N/2)], PSD3[1: int(N/2)]-150, label='TX5-RX3')
    plt.plot(tau4[1:int(N/2)], PSD4[1: int(N/2)]-200, label='TX5-RX4')
    plt.plot(tau0[1:int(N/2)], PSD0[1: int(N/2)], label='Through')

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




def plot_1by1(tau, PSD, data_name, out_dir):
    plt.plot(tau[1:int(len(tau)/2)], PSD[1:int(len(PSD)/2)])

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
plot_1by1(tau1, PSD1, 'TX5-RX1.csv', out_dir1)
plot_1by1(tau2, PSD2, 'TX5-RX2.csv', out_dir2)
plot_1by1(tau3, PSD3, 'TX5-RX3.csv', out_dir3)
plot_1by1(tau4, PSD4, 'TX5-RX4.csv', out_dir4)
plot_1by1(tau0, PSD0, 'Through.csv', out_dir0)
"""
