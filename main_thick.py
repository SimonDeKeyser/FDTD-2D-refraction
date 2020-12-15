import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.05
kd = 5
kdmin = 0.1
kdmax = 10
CFL = 0.8  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 500
obj = 'thick' # Object to simulate
A = 10
sigma = 1e-5
source = (A,sigma) #source parameters: (A,Sigma)

# RUN SIMULATION---------------------------------------------
thick = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,obj,animation=False)
thick.run()
thick.save()

# LOAD SIMULATION----------------------------------------
recorders, source = thick.load()

# TIME_FFT SUMMARY----------------------------------------
thick.time_fft_summary(1,recorders=recorders,source=source) #plot a summary

# ANALYTICAL COMPARISON-----------------------------------
TF_1 = thick.TF_FDTD(1,recorders=recorders,source=source)
TF_ana_1 = thick.TF_ANA(1)
thick.FDTD_ana_comparison(1,TF_1,TF_ana_1,rel=False) #plot a FDTD/analytical comparison

