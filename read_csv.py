#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd


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
one_file('TX5-RX1.csv')