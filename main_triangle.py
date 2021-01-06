import numpy as np
### If CONFORMAL: change SimulationEngine to Conformal here: ###
from SimulationEngine import FDTD
import matplotlib.pyplot as plt

# PARAMETERS----------------------------------------------
c = 340  # geluidssnelheid - speed of sound (wave speed)
dx = 0.05
#dx= 0.04
#dx = 0.03
kd = 5
kdmin = 1
kdmax = 9
CFL = 0.5
dt = CFL / (c * np.sqrt((1 / dx ** 2) + (1 / dx ** 2)))  # time step
nt = 840
#nt = 900
#nt = 1020
obj = 'triangle' # Object to simulate
A = 10
sigma = (1e-5)
source = (A,sigma) #source parameters: (A,Sigma)

# RUN SIMULATION---------------------------------------------
triangle = FDTD(dx,kd,dt,nt,source,kdmin,kdmax,obj,animation=False)
triangle.run()
triangle.save()

# LOAD SIMULATION----------------------------------------
recorders, source = triangle.load()

recorder_number = 1
# TIME_FFT SUMMARY----------------------------------------
triangle.time_fft_summary(recorder_number,recorders=recorders,source=source) #plot a summary

# ANALYTICAL COMPARISON-----------------------------------
TF_1 = triangle.TF_FDTD(recorder_number,recorders=recorders,source=source)
TF_ana_1 = triangle.TF_ANA(recorder_number)
triangle.FDTD_ana_comparison(recorder_number,TF_1,TF_ana_1,relative=True) #plot a FDTD/analytical comparison

print('RMS: {}'.format(np.sqrt(np.mean(np.square(np.abs(TF_1)-np.abs(TF_ana_1))/np.abs(TF_ana_1)))))
