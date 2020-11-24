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
obj = 'triangle' # Object to simulate

# SIMULATION----------------------------------------------
triangle = FDTD(dx,kd,dt,nt,obj,animation=False)
triangle.run()
freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=False)
freefield.run()

# PLOT----------------------------------------------------
triangle.plot_recorders()
freefield.plot_recorders()

# ANALYTICAL COMPARISON------------------------------------
triangle.bront = np.reshape(triangle.bront, (nt, )) #signal.TransferFunction requires vector
TF_1 = signal.TransferFunction(triangle.recorder1-freefield.recorder1,triangle.bront,dt=dt)
TF_2 = signal.TransferFunction(triangle.recorder2-freefield.recorder2,triangle.bront,dt=dt)
TF_3 = signal.TransferFunction(triangle.recorder3-freefield.recorder3,triangle.bront,dt=dt)