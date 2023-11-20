import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
import pandas as pd
from tqdm import tqdm

# Read csv data
data_51 = pd.read_csv('Ascan/TX5-RX1.csv/tau_t_Output.csv', header=None, skiprows=19)
data_52 = pd.read_csv('Ascan/TX5-RX2.csv/tau_t_Output.csv', header=None, skiprows=19)
data_53 = pd.read_csv('Ascan/TX5-RX3.csv/tau_t_Output.csv', header=None, skiprows=19)
data_54 = pd.read_csv('Ascan/TX5-RX4.csv/tau_t_Output.csv', header=None, skiprows=19)

# output data
Amp_51 = list(data_51[1]) # IF output [V]
Amp_52 = data_52[1] # IF output [V]
Amp_53 = data_53[1] # IF output [V]
Amp_54 = data_54[1] # IF output [V]


# distance between TX and RX
L_51 = 0.2 * 4 # [m]
L_52 = 0.2 * 3 # [m]
L_53 = 0.2 * 2 # [m]
L_54 = 0.2 * 1 # [m]


delta_tau = 0.01e-9 # time resolution [s]
tau = np.arange(0, 20e-9, delta_tau) # time [s]
V_RMS = np.arange(0.01, 1.01, 0.01) # list of RMS voltage, generalized in C = 3e8 (speed of light)

corr = np.zeros((len(tau), len(V_RMS)))


for i in tqdm(range(len(V_RMS))):
    v = V_RMS[i]* 3e8
    for j in range(len(tau)):
        t = tau[j]
        # calculate delay time and convert to index number
        delay_time_51 = np.sqrt(t**2 + (L_51 / v)**2)
        delay_time_52 = np.sqrt(t**2 + (L_52 / v)**2)
        delay_time_53 = np.sqrt(t**2 + (L_53 / v)**2)
        delay_time_54 = np.sqrt(t**2 + (L_54 / v)**2)

        outvalue_51 = Amp_51.index(delay_time_51)
        outvalue_52 = Amp_52.index(delay_time_52)
        outvalue_53 = Amp_53.index(delay_time_53)
        outvalue_54 = Amp_54.index(delay_time_54)

        corr[j, i] = outvalue_51 * outvalue_52 + outvalue_51 * outvalue_53 + outvalue_51 * outvalue_54 \
            + outvalue_52 * outvalue_53 + outvalue_52 * outvalue_54 \
            + outvalue_53 * outvalue_54
        
        #print(t, v, corr[v, t])

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
plt.imshow(corr,
            extent=[0, len(V_RMS)*0.01, len(tau)*0.1, 0], 
            cmap='inferno', aspect='auto')

plt.xlabel('RMS Velocity [/c]')
plt.ylabel('tau [ns]')
plt.title('Correlation')

# set clorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label = 'correration')


plt.show()
