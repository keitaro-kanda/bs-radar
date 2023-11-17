import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
import pandas as pd
from tqdm import tqdm

# Read csv data
data_51 = pd.read_csv('TX5-RX1.csv', header=None, skiprows=19)
data_52 = pd.read_csv('TX5-RX2.csv', header=None, skiprows=19)
data_53 = pd.read_csv('TX5-RX3.csv', header=None, skiprows=19)
data_54 = pd.read_csv('TX5-RX4.csv', header=None, skiprows=19)

# output data
out_51 = data_51[2] # IF output [V]
out_52 = data_52[2] # IF output [V]
out_53 = data_53[2] # IF output [V]
out_54 = data_54[2] # IF output [V]


# distance between TX and RX
L_51 = 0.2 * 4 # [m]
L_52 = 0.2 * 3 # [m]
L_53 = 0.2 * 2 # [m]
L_54 = 0.2 * 1 # [m]


delta_t = 2 / 32768
Time = np.arange(0, 2, 0.01) # time [s]
V_RMS = np.arange(0.01, 1.01, 0.01) # list of RMS voltage, generalized in C = 3e8 (speed of light)

corr = np.zeros((len(Time), len(V_RMS)))


for i in tqdm(range(len(V_RMS))):
    v = V_RMS[i]* 3e8
    for j in range(len(Time)):
        t = Time[j]
        # calculate delay time and convert to index number
        delay_time_51 = np.int(np.sqrt(t**2 + (L_51 / v)**2) / delta_t)
        delay_time_52 = np.int(np.sqrt(t**2 + (L_52 / v)**2) / delta_t)
        delay_time_53 = np.int(np.sqrt(t**2 + (L_53 / v)**2) / delta_t)
        delay_time_54 = np.int(np.sqrt(t**2 + (L_54 / v)**2) / delta_t)
        print(delay_time_51, delay_time_52, delay_time_53, delay_time_54)

        outvalue_51 = out_51[delay_time_51]
        outvalue_52 = out_52[delay_time_52]
        outvalue_53 = out_53[delay_time_53]
        outvalue_54 = out_54[delay_time_54]

        corr[j, i] = outvalue_51 * outvalue_52 + outvalue_51 * outvalue_53 + outvalue_51 * outvalue_54 \
            + outvalue_52 * outvalue_53 + outvalue_52 * outvalue_54 \
            + outvalue_53 * outvalue_54
        
        #print(t, v, corr[v, t])

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
plt.imshow(corr,
            #extent=[0, len(V_RMS)*0.01, len(Time)*0.01, 0], 
            cmap='hot', aspect='auto')

plt.xlabel('RMS Velocity [/c]')
plt.ylabel('Time [s]')
plt.title('Correlation')

# set clorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label = 'correration')


plt.show()
