import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from FDTD import *

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx=0.05
kd=1
CFL = 0.8  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 600
obj = 'thin' # Object to simulate

# SIMULATION----------------------------------------------
bront, rec = Simulation(dx,kd,dt,nt,obj,plot=True)
bront, recfree = Simulation(dx,kd,dt,nt,'freefield_'+obj)

# PLOT----------------------------------------------------
t = np.arange(nt)*dt
#plt.plot(t,bront,label='Source')
plt.plot(t,rec[0],label='Recorder 1')
plt.plot(t,rec[1],label='Recorder 2')
plt.plot(t,rec[2],label='Recorder 3')
plt.title('Recorded wave vs time, {} object case'.format(obj))
plt.xlabel('Time [s]')
plt.ylabel('Pressure [N/m^2]')
plt.legend()
plt.grid()
plt.show()

# ANALYTICAL COMPARISON------------------------------------
bront = np.reshape(bront, (nt, )) #signal.TransferFunction requires vector
TF_1 = signal.TransferFunction(rec[0]-recfree[0],bront,dt=dt)
TF_2 = signal.TransferFunction(rec[1]-recfree[1],bront,dt=dt)
TF_3 = signal.TransferFunction(rec[2]-recfree[2],bront,dt=dt)