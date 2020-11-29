import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt
from scipy.special import hankel2, hankel1

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.2
kd = 3
d=2
nd = d/dx
CFL = 1  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 400
obj = 'thin' # Object to simulate

# SIMULATION----------------------------------------------
#thin = FDTD(dx,kd,dt,nt,obj,animation=False)
#thin.run()
freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=True)
freefield.run()
recorders = np.load('freefield_thin_recorders_kd={}.npy'.format(kd)) #freefield was saved (large simulation)
source = np.load('freefield_thin_source_kd={}.npy'.format(kd))

# PLOT----------------------------------------------------
time = np.arange(nt)*dt
FFT_source, k_vec = freefield.fft(source,8000)
FFT_1, _ = freefield.fft(recorders[0],8000)
FFT_2, _ = freefield.fft(recorders[1],8000)
freefield.plot_time_fft(time,source,recorders[0],k_vec,FFT_source,FFT_1) #plot a summary

# ANALYTICAL COMPARISON------------------------------------
TF_1, k_vec = freefield.TF_SIM(recorder_number=1,recorders=recorders,source=source, plot= True)
r1 = np.sqrt((2)**2+(2/5)**2)*d #distance to first recorder
TF_ana = hankel1(0,k_vec*r1)/(FFT_source)
freefield.plot_FDTD_ana(k_vec,TF_ana,TF_1) #plot a FDTD/analytical comparison

## WANHOPIGE CONTROLE MET PC LES 1
r2 = np.sqrt((3)**2+(2/5)**2)*d #distance to second recorder
Averhouding_FDTD = np.abs(FFT_1/FFT_2)
Averhouding_theo = np.abs(hankel1(0,k_vec*r1)/hankel1(0,k_vec*r2))
Averhoudingrel = Averhouding_FDTD/Averhouding_theo
Averhouding = 1+((Averhoudingrel-1)/nd)

plt.plot(2*np.pi/(k_vec*dx),Averhouding)
plt.title('ratio fft recorder1/recorder2 vs analytical')
plt.ylabel('ratio')
plt.xlabel('number of cells per wavelength')
plt.show()