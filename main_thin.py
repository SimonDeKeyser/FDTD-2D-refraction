import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.2
kd = 5
kdmin = kd*0.2
kdmax = kd*1.7
CFL = 0.8  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 800
obj = 'thin' # Object to simulate
A = 10
sigma = 1e-5
source = (A,sigma) #source parameters: (A,Sigma)

# RUN SIMULATION---------------------------------------------
thin = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,obj,animation=False)
free = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,'freefield_'+obj,animation=False)
thin.run()
free.run()

# LOAD SIMULATION----------------------------------------
recorders = np.load('{}_recorders_kd={}.npy'.format(obj, kd)) #loading saved recorders, MAKE SURE YOU USED THE SAME PARAMETERS
source = np.load('{}_source_kd={}.npy'.format(obj, kd))
free_recorders = np.load('{}_recorders_kd={}.npy'.format('freefield_'+obj, kd)) #loading saved recorders, MAKE SURE YOU USED THE SAME PARAMETERS
free_source = np.load('{}_source_kd={}.npy'.format('freefield_'+obj, kd))

# TIME_FFT SUMMARY----------------------------------------
thin.time_fft_summary(1,recorders=recorders,source=source) #plot a summary

# ANALYTICAL COMPARISON-----------------------------------
TF_1 = thin.TF_FDTD(1,recorders=recorders,source=source)
TF_free = free.TF_FDTD(1,recorders=free_recorders,source=free_source)
TF_ana_1 = thin.TF_ANA(1)
thin.FDTD_ana_comparison(TF_1-TF_free,TF_ana_1) #plot a FDTD/analytical comparison

