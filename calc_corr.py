import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
import pandas as pd
from tqdm import tqdm


# =====Read A-scan data=====
def read_Ascan(i):
    file_path = 'Ascan/TX5-RX' + str(i)  + '/tau_ASD_PSD.csv'
    data = pd.read_csv(file_path, header=None, skiprows=1)
    delay_time = data[0] # delay time [s]
    AS = data[1]  # Amplitude Spectrum, [V]
    ASD = data[2] # Amplitude Spectrum Density, [V/√Hz]
    PSD = data[3] # Power Spectrum Density, [dB/Hz]
    delay_time = delay_time - delay_time[np.argmax(PSD)] # tau_0をt=0にする
    return delay_time, AS, ASD, PSD

tau1, AS1, ASD1, PSD1 = read_Ascan(1) # pandas, series型
tau2, AS2, ASD2, PSD2 = read_Ascan(2)
tau3, AS3, ASD3, PSD3 = read_Ascan(3)
tau4, AS4, ASD4, PSD4 = read_Ascan(4)


# =====distance between TX and RX=====
L_1 = 0.2 * 4 # [m]
L_2 = 0.2 * 3 # [m]
L_3 = 0.2 * 2 # [m]
L_4 = 0.2 * 1 # [m]


# =====caluculate delta tau=====
fs = 16000 # sampling frequency [Hz]
N = 32768 # data size
sweep_rate = 0.9e9 # [Hz/s]
delta_tau = fs / N / sweep_rate  # resolution of tau [s]


# =====make axis lists=====
max_tau = 20 # [ns]
delta_t = 0.1 # [ns]
tau = np.arange(0, max_tau * 1e-9, delta_t * 1e-9) # time series [s]

V_RMS = np.arange(0.01, 1.01, 0.01) # RMS velocity series, normalized by c

corr = np.zeros((len(tau), len(V_RMS)))


for i in tqdm(range(len(V_RMS))):
    v = V_RMS[i]* 3e8 #  [m/s]
    for j in range(len(tau)):
        t = tau[j] # [s]
        # calculate delay time and convert to index number
        delay_time1 = np.sqrt(t**2 + (L_1 / v)**2) + 44.48e-9
        delay_time2 = np.sqrt(t**2 + (L_2 / v)**2) + 44.48e-9
        delay_time3 = np.sqrt(t**2 + (L_3 / v)**2) + 44.48e-9
        delay_time4 = np.sqrt(t**2 + (L_4 / v)**2) + 44.48e-9

        
        f1 = AS1[(delay_time1 - delta_tau/2 < tau1) & (tau1 < delay_time1 + delta_tau/2)].values[0]
        f2 = AS2[(delay_time2 - delta_tau/2 < tau2) & (tau2 < delay_time2 + delta_tau/2)].values[0]
        f3 = AS3[(delay_time3 - delta_tau/2 < tau3) & (tau3 < delay_time3 + delta_tau/2)].values[0]
        f4 = AS4[(delay_time4 - delta_tau/2 < tau4) & (tau4 < delay_time4 + delta_tau/2)].values[0]
        #print(type(f1), type(f2), type(f3), type(f4))
        #print(f1, f2, f3, f4)


        corr[j, i] = f1 * f2 + f1 * f3 + f1 * f4 \
            + f2 * f3 + f2 * f4 \
            + f3 * f4
np.savetxt('corr/corr.txt', corr, delimiter=',', fmt='%s')

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
plt.imshow(corr,
            extent=[0, len(V_RMS)*0.01, max_tau, 0], 
            cmap='inferno', aspect='auto')

plt.xlabel('RMS Velocity [/c]')
plt.ylabel('tau (2-way travel time) [ns]')
plt.title('Correlation')

# set clorbar
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label = 'correration')

plt.savefig('corr/corr.png', dpi=300)
plt.show()
