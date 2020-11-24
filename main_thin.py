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
obj = 'thin' # Object to simulate

# SIMULATION----------------------------------------------
thin = FDTD(dx,kd,dt,nt,obj,animation=False)
thin.run()
freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=False)
freefield.run()

# PLOT----------------------------------------------------
thin.plot_recorders()
freefield.plot_recorders()

# ANALYTICAL COMPARISON------------------------------------
thin.bront = np.reshape(thin.bront, (nt, )) #signal.TransferFunction requires vector
TF_1 = signal.TransferFunction(thin.recorder1-freefield.recorder1,thin.bront,dt=dt)
TF_2 = signal.TransferFunction(thin.recorder2-freefield.recorder2,thin.bront,dt=dt)
TF_3 = signal.TransferFunction(thin.recorder3-freefield.recorder3,thin.bront,dt=dt)