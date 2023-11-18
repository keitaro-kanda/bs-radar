#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import tight_layout


# Read and plot csv data 1 by 1
def one_file(path):
    # Read in the data from the csv file
    file_path = path
    file_name = file_path.split('.')[0]
    data = pd.read_csv(file_path, header=None, skiprows=19)

    time = data[0] # Time [s]
    input = data[1] # Voltage applied to VCO [V]
    output = data[2] # IF output [V]


    # Plot the data
    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].plot(time, input)
    #ax[0].set_title('Input', size = 16)
    ax[0].set_ylabel('Input Voltage [V]', size = 14)

    ax[1].plot(time, output)
    #ax[1].set_title('Output', fontsize = 16)
    ax[1].set_xlabel('Time [s]', size = 14)
    ax[1].set_ylabel('Output Voltage [V]', size = 14)
    ax[1].set_ylim(0.18, 0.30)

    plt.savefig('tV_plot/'+file_name + '.png', dpi=300)
    plt.show()
#one_file('TX5-RX1.csv')


# Rea and plot all csv data 
def all_file():
    
    data_51 = pd.read_csv('TX5-RX1.csv', header=None, skiprows=19)
    data_52 = pd.read_csv('TX5-RX2.csv', header=None, skiprows=19)
    data_53 = pd.read_csv('TX5-RX3.csv', header=None, skiprows=19)
    data_54 = pd.read_csv('TX5-RX4.csv', header=None, skiprows=19)
    data_Th = pd.read_csv('Through.csv', header=None, skiprows=19)

    Time = data_51[0] # Time [s]
    Input_51 = data_51[1] # Voltage applied to VCO [V]
    Output_51 = data_51[2] # IF output [V]
    Output_52 = data_52[2] # IF output [V]
    Output_53 = data_53[2] # IF output [V]
    Output_54 = data_54[2] # IF output [V]
    Output_th = data_Th[2] # IF output [V]


    # Plot the data
    fig, ax = plt.subplots(6, 1, sharex='all', tight_layout=True, figsize=(8, 10))

    ax[0].plot(Time, Input_51)
    ax[0].set_title('Input', size = 16)

    ax[1].plot(Time, Output_51)
    ax[1].set_title('Output TX5-RX1', fontsize = 16)
    ax[1].set_ylim(0.17, 0.31)

    ax[2].plot(Time, Output_52)
    ax[2].set_title('Output TX5-RX2', fontsize = 16)
    ax[2].set_ylim(0.17, 0.31)

    ax[3].plot(Time, Output_53)
    ax[3].set_title('Output TX5-RX3', fontsize = 16)
    ax[3].set_ylim(0.17, 0.31)

    ax[4].plot(Time, Output_54)
    ax[4].set_title('Output TX5-RX4', fontsize = 16)
    ax[4].set_ylim(0.17, 0.31)

    ax[5].plot(Time, Output_th)
    ax[5].set_title('Output Through', fontsize = 16)
    ax[5].set_ylim(0.17, 0.31)


    #fig.supxlabel('Time [s]', size = 14)
    #fig.supylabel('Voltage [V]', size = 14)


    plt.savefig('tV_plot/all.png', dpi=300)
    plt.show()
all_file()
