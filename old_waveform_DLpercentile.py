
import numpy as np
import h5py
from figaro.load import load_single_event
from figaro.cosmology import CosmologicalParameters

z_c=0.438
DL_em = CosmologicalParameters(0.674,0.315,0.685,-1.,0.).LuminosityDistance_double(z_c)

#LVK old waveform 
filepath='old_waveform/GW190521_posterior_samples.h5'

samples, name = load_single_event(filepath, par = ['luminosity_distance'], waveform = 'imr')
samples = np.sort(samples.flatten())
perc = np.argmin(abs(samples - DL_em))/len(samples)
print(perc*100)

