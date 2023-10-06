
import numpy as np
import h5py
from figaro.load import load_density
from figaro.cosmology import CosmologicalParameters

z_c=0.438
DL_em = CosmologicalParameters(0.674,0.315,0.685,-1.,0.).LuminosityDistance_double(z_c)

D_L=np.linspace(1,10000,202)[1:-1]

#LVK old waveform 
filepath='old_waveform/GW190521_posterior_samples.h5'

draws= load_density(filepath)
draws_pdf = np.mean([d.pdf(D_L.T) for d in draws], axis = 0).reshape(len(D_L)) 
perc = np.sum(draws_pdf[D_L<DL_em]*(D_L[1]-D_L[0]))
print(perc*100)

