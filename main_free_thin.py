import numpy as np
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx= 0.04
kd = 5
kdmin = 0.1
kdmax = 10
CFL = 1  # Courant number
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 200
obj = 'freefield_thin' # Object to simulate
A = 1
sigma = 1e-5/3
source = (A,sigma) #source parameters: (A,Sigma)

# RUN SIMULATION---------------------------------------------
free_thin = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,obj,animation=False)
free_thin.run()
free_thin.save()

# LOAD SIMULATION----------------------------------------
recorders, source = free_thin.load()

recorder_number = 1
# TIME_FFT SUMMARY----------------------------------------
free_thin.time_fft_summary(recorder_number,recorders=recorders,source=source) #plot a summary

# ANALYTICAL COMPARISON-----------------------------------
TF_1 = free_thin.TF_FDTD(recorder_number,recorders=recorders,source=source)
TF_ana_1 = free_thin.TF_ANA(recorder_number)
free_thin.FDTD_ana_comparison(recorder_number,TF_1,TF_ana_1,relative=False) #plot a FDTD/analytical comparison

print('RMS: {}'.format(np.sqrt(np.mean(np.square(np.abs(TF_1)-np.abs(TF_ana_1))/np.abs(TF_ana_1)))))