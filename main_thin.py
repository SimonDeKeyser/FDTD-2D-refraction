import numpy as np
from SimulationEngine import FDTD

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
#freefield = FDTD(dx,kd,dt,nt,'freefield_'+obj,animation=False)
#freefield.run()

# PLOT----------------------------------------------------
thin.plot_recorders()
thin.plot_source()

# ANALYTICAL COMPARISON------------------------------------
TF_thin = thin.TF_SIM(recorder_number=1,plot=True)
TF_ana = thin.TF_ANA(recorder_number=1,plot=True)