import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.1
kd = 5
kdmin = kd*0.2
kdmax = kd*1.7
CFL = 1  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 400
obj = 'freefield_thin' # Object to simulate
A = 10
sigma = 1e-5
source = (A,sigma) #source parameters: (A,Sigma)

# RUN SIMULATION---------------------------------------------
free_thin = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,obj,animation=False)
#free_thin.run()

# LOAD SIMULATION----------------------------------------
recorders = np.load('{}_recorders_kd={}.npy'.format(obj, kd)) #loading saved recorders, MAKE SURE YOU USED THE SAME PARAMETERS
source = np.load('{}_source_kd={}.npy'.format(obj, kd))

# TIME_FFT SUMMARY----------------------------------------
free_thin.time_fft_summary(1,recorders=recorders,source=source) #plot a summary

# ANALYTICAL COMPARISON-----------------------------------
TF_1 = free_thin.TF_FDTD(1,recorders=recorders,source=source)
TF_ana_1 = free_thin.TF_ANA(1)
free_thin.FDTD_ana_comparison(TF_1,TF_ana_1) #plot a FDTD/analytical comparison
