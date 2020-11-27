import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx=0.05
kd=5
CFL = 0.8  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 1000
obj = 'thin' # Object to simulate

# SIMULATION----------------------------------------------
thin = FDTD(dx,kd,dt,nt,obj,animation=False)
thin.run()
freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=False)
#freefield.run()
recorders = np.load('freefield_thin_recorders_kd=5.npy') #freefield was saved (large simulation)
source = np.load('freefield_thin_source_kd=5.npy')

# PLOT----------------------------------------------------
thin.plot_recorders()
thin.plot_source()
freefield.plot_recorders(recorders=recorders)

# ANALYTICAL COMPARISON------------------------------------
TF_ana, k_vec = thin.TF_ANA(recorder_number=1)
TF_freefield, k_fft = freefield.TF_SIM(recorder_number=1,recorders=recorders,source=source)
TF_thin, k_fft = thin.TF_SIM(recorder_number=1)

TF = TF_thin-TF_freefield
Amp = np.abs(TF)
Phase = np.unwrap(np.angle(TF)) 
Amp_ana = np.abs(TF_ana)
Phase_ana = np.unwrap(np.angle(TF_ana)) 
ax1 = plt.subplot(2,1,1)
ax1.plot(k_vec,Amp_ana,label='Analytical')
ax1.plot(k_fft,Amp,label='Simulated')
ax1.set_ylabel('Amplitude')
ax1.set_title('Transfer function (freefield subtracted), thin sheet case')
ax1.grid()
ax2 = plt.subplot(2,1,2)
ax2.plot(k_vec,Phase_ana,label='Analytical')
ax2.plot(k_fft,Phase,label='Simulated')
ax2.set_xlabel('kd')
ax2.set_ylabel('Phase')
ax2.grid()
plt.legend()
plt.show()
