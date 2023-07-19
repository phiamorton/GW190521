from os import read
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import h5py
from figaro.load import load_single_event
from GW190521_raynest import redshift_model
from figaro.load import load_density
import raynest
import raynest.model
from figaro.cosmology import CosmologicalParameters

#LVK distribution:
samples_in, name = load_single_event('GW190521.h5',par= ['mc_detect','luminosity_distance']) 
#detector frame M_C
#this should be the same because both models should predict what we observe


#redshifted model from conditioned draws:
dpgmm_file = 'conditioned_density_draws.pkl' #non-redshifted M_c ?? OR IS IT
#the conditional distribution (based on EM sky location)
#z_c from EM counterpart candidate https://arxiv.org/pdf/2006.14122.pdf at ~2500 Mpc
z_c = 0.438
GW_posteriors = load_density(dpgmm_file)

mymodel= redshift_model(z_c, GW_posteriors)

#post process
with h5py.File('inference/raynest.h5', 'r') as f:
        post = np.array(f['combined']['posterior_samples'])


samples_out = np.column_stack([post[lab] for lab in mymodel.names])

omega = CosmologicalParameters(0.674, 0.315, 0.685, -1., 0.)
DL_em = omega.LuminosityDistance_double(z_c)
#print(DL_em)
reconstruction= samples_out[:,[0,1]]
    #reconstruction[:,0]=np.exp(reconstruction[:,0]) #use if samples of r are log
r=np.exp(samples_out[:,0])
vel=1./np.sqrt(2*(r-1))  #the magnitude at a given distance from SMBH
vel_LoS = vel * samples_out[:,2] #* np.cos(samples[:,3]) * np.cos(samples[:,4]) #Ive created a monster :((
    #gamma=lorentz factor
gamma = 1./np.sqrt(1 - vel**2)

    #z_rel (r, angles)
    #make bounds on angle 0 to 2pi, redshifted should be when v_LoS is negative (theta=pi)
z_rel = gamma * (1 + vel_LoS) - 1

    #z_grav (r)
z_grav = 1./np.sqrt(1 - 1./r) - 1 
    #D_L eff (z_c, z_rel, z_grav, D_L)
D_eff = (1+z_rel)**2 * (1+z_grav) * DL_em 

M_eff = (1+z_c) * (1 + z_rel) * (1 + z_grav) * samples_out[:,1]

# samples_out[:,1]=M_eff 
samples_out = np.array([M_eff, D_eff]).T

#add in conditioned dist. 
from figaro.plot import plot_multidim 
from figaro.marginal import condition

#LVK draws pkl file, unconditioned on EM sky location
filepath= 'draws_GW190521.pkl'

ra_EM, dec_EM = 192.42625 , 34.8247

ra_EM_rad= ra_EM/360*np.pi*2
dec_EM_rad=dec_EM/180*np.pi 

draws = load_density(filepath)

#conditioned draws we are drawing from for comparison
conditioned_draws = condition(draws,[ra_EM_rad,dec_EM_rad], [2,3], norm=True, filter=True, tol=1e-3)

#want to plot redshift model M_c eff and D_L posterior
#vs conditioned LVK dist.
#vs LVK non conditioned dist. 
#fixed cosmology for all

# fig=plot_multidim(conditioned_draws, samples = samples_out[:,[1,0]],labels = [ 'M_c effective ',' D_L effective']) 
fig=plot_multidim(conditioned_draws, labels = ['M_c', 'D_{effective}'], units=['M_\\odot', 'Mpc'])
fig=corner(samples_out, color='magenta', fig=fig, label='redshift model', hist_kwargs ={'density':True, 'label':'redshifted model'})
fig=corner(samples_in[:,[0,1]], color='grey', fig=fig, label='LVK', plot_density=False, hist_kwargs ={'density':True, 'label':'LVK model'})

#agh making legends is stupid!!
fig.axes[1].legend(*fig.axes[0].get_legend_handles_labels(), loc='center', frameon=False)

fig.savefig('overplot.pdf')

