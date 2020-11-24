import numpy as np
from scipy import signal
from SimulationEngine import FDTD

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx=0.05
kd=1
CFL = 0.8  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 600
obj = 'thick' # Object to simulate

# SIMULATION----------------------------------------------
thick = FDTD(dx,kd,dt,nt,obj,animation=False)
thick.run()
freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=False)
freefield.run()

# PLOT----------------------------------------------------
thick.plot_recorders()
freefield.plot_recorders()

# ANALYTICAL COMPARISON------------------------------------
thick.bront = np.reshape(thick.bront, (nt, )) #signal.TransferFunction requires vector
TF_1 = signal.TransferFunction(thick.recorder1-freefield.recorder1,thick.bront,dt=dt)
TF_1 = signal.TransferFunction(thick.recorder2-freefield.recorder2,thick.bront,dt=dt)
TF_1 = signal.TransferFunction(thick.recorder3-freefield.recorder3,thick.bront,dt=dt)