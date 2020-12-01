import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt
from scipy.special import hankel2, hankel1

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.1
kd = 5
d = 1
nd = d/dx
CFL = 1  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 600
obj = 'thin' # Object to simulate
A = 10
sigma = 5e-5
source = (A,sigma) #source parameters: (A,Sigma)

# SIMULATION----------------------------------------------
#thin = FDTD(dx,kd,dt,nt,obj,animation=False)

#thin.run()
freefield = FDTD(dx,kd,dt,nt,source,'freefield_'+obj,animation=False)
freefield.run()
recorders = np.load('freefield_thin_recorders_kd={}.npy'.format(kd)) #freefield was saved (large simulation)
source = np.load('freefield_thin_source_kd={}.npy'.format(kd))
time = np.arange(nt)*dt

# PLOT----------------------------------------------------

FFT_source, k_vec = freefield.fft(source,9000)
FFT_1, _ = freefield.fft(recorders[0],9000)
FFT_2, _ = freefield.fft(recorders[1],9000)
freefield.plot_time_fft(time,source,recorders[0],k_vec,FFT_source,FFT_1) #plot a summary

# ANALYTICAL COMPARISON------------------------------------
TF_1, k_vec = freefield.TF_SIM(recorder_number=1,recorders=recorders,source=source)
r1 = np.sqrt((2)**2+(2/5)**2)*d #distance to first recorder
TF_ana = -1j*hankel2(0,k_vec*r1)*k_vec/(A*4)
freefield.plot_FDTD_ana(k_vec,TF_ana,TF_1) #plot a FDTD/analytical comparison

## CONTROLE PC LES 1
r2 = np.sqrt((3)**2+(2/5)**2)*d #distance to second recorder
Averhouding_FDTD = np.abs(FFT_1/FFT_2)
Averhouding_theo = np.abs(hankel2(0,k_vec*r1)/hankel2(0,k_vec*r2))
Averhoudingrel = Averhouding_FDTD/Averhouding_theo
Averhouding = 1+((Averhoudingrel-1)/nd)

plt.plot(2*np.pi/(k_vec*dx),Averhouding)
plt.title('ratio FDTD/ana FFT at recorders 1 and 2')
plt.ylabel('ratio')
plt.grid()
plt.xlabel('number of cells per wavelength')
plt.ylim([0.99, 1.01])
plt.show()


plt.plot(k_vec*d,abs(FFT_1),label='FFT recorder 1')
plt.plot(k_vec*d,abs(TF_ana*FFT_source),label='TF_ana * FFT_source')
plt.title('fft recorder1')
plt.xlabel('kd')
plt.ylabel('amplitude')
plt.legend()
plt.show()