import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
from corner import corner
import h5py
import pickle

from scipy.special import logsumexp
from numba import njit

from figaro.cosmology import CosmologicalParameters
from figaro.load import load_density
from figaro.likelihood import logsumexp_jit
from figaro.plot import plot_multidim 

from priors import rad_prior, pl_peak_LVK, pl_peak_no_tapering, logprior_luminosity_distance 

# dpgmm_file = 'primarymass/marginalized_density_draws_M1_and_DL.pkl' #detector frame M_1, not conditioned

# GW_posteriors = load_density(dpgmm_file)

dpgmm_file= 'Marginalized_interpolation_allsky.pkl'
with open(dpgmm_file, 'rb') as f:
    GW_posteriors = pickle.load(f)

def GW_post(M,DL):
    return GW_posteriors(M,DL)

class noEM_model_plpk(raynest.model.Model):

    def __init__(self, draws):
        super(noEM_model_plpk,self).__init__()
        # self.draws=draws
        # self.N_draws = len(self.draws)
        # self.ones    = np.ones(self.N_draws)
        self.omega   = CosmologicalParameters(0.674,0.315,0.685,-1.,0.)
        
        self.names= ['M_1',# source frame primary mass
                     'z_c'] #cosmological redshift
                
        self.bounds =[ [0.,300.], [0,2] ]
        
    def log_prior(self,x):

        logp=super(noEM_model_plpk,self).log_prior(x)

        if np.isfinite(logp):    
            logp_M_c = pl_peak_LVK(x['M_1']) #power law +peak 
            logp_z   = np.log(self.omega.ComovingVolumeElement_double(x['z_c'])) #unifrom in comoving volume 
            return logp_M_c +logp_z
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL = np.float64(self.omega.LuminosityDistance_double(x['z_c']))
        
        M_eff = (1+x['z_c'])* x['M_1'] #primary mass with cosmological redshift= M_eff
        #print(type(M_eff))
        #print(type(DL))
        logl = GW_post(M_eff, DL)
        logl -= logprior_luminosity_distance(DL)
        return logl


noEM_plpk_model= noEM_model_plpk(GW_posteriors)

postprocess=False
if not postprocess:
    nest_noEM_plpk = raynest.raynest(noEM_plpk_model, verbose=2, nnest=1, nensemble=1, nlive=2000, maxmcmc=5000, output = 'inference_noEM_plpk/')
    nest_noEM_plpk.run(corner = True)
    post_noEM_plpk = nest_noEM_plpk.posterior_samples.ravel()
else:
    with h5py.File('inference_noEM_plpk/raynest.h5', 'r') as f:
        no_EM_plpk_logZ= np.array(f['combined']['logZ'])
        post_noEM_plpk = np.array(f['combined']['posterior_samples'])

samples1 = np.column_stack([post_noEM_plpk[lab1] for lab1 in noEM_plpk_model.names])
fig = corner(samples1, labels = ['$M_1$', 'z_c'], truths = [85.5, None])
fig.savefig('inference_noEM_plpk/noEM_plpk_posterior.pdf', bbox_inches = 'tight')


class noEM_model_plpk_no_tapering(raynest.model.Model):

    def __init__(self, draws):
        super(noEM_model_plpk_no_tapering,self).__init__()
        self.draws=draws
        # self.N_draws = len(self.draws)
        # self.ones    = np.ones(self.N_draws)
        self.omega   = CosmologicalParameters(0.674,0.315,0.685,-1.,0.)
        
        self.names= ['M_1',# M_C source frame chirp mass
                     'z_c'] 
                
        self.bounds =[ [0.,300.], [0,2] ]
        
    def log_prior(self,x):

        logp=super(noEM_model_plpk_no_tapering,self).log_prior(x)

        if np.isfinite(logp):    
            logp_M_c = pl_peak_no_tapering(x['M_1']) #power law +peak  but without the tapering
            logp_z   = np.log(self.omega.ComovingVolumeElement_double(x['z_c'])) #unifrom in comoving volume 
            return logp_M_c + logp_z
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL = self.omega.LuminosityDistance_double(x['z_c'])
        
        M_eff = (1+x['z_c'])* x['M_1'] #chirp mass with cosmological redshift= M_eff

        logl = GW_post(M_eff, DL) #one draw
        logl -= logprior_luminosity_distance(DL) #remove GW prior
        return logl

noEM_plpk_no_tapering_model= noEM_model_plpk_no_tapering(GW_posteriors)

postprocess=False
if not postprocess:
    nest_noEM_plpk_no_tapering = raynest.raynest(noEM_plpk_no_tapering_model, verbose=2, nnest=1, nensemble=1, nlive=2000, maxmcmc=5000, output = 'inference_no_tapering/')
    nest_noEM_plpk_no_tapering.run(corner = True)
    post_noEM_plpk_no_tapering = nest_noEM_plpk_no_tapering.posterior_samples.ravel()
else:
    with h5py.File('inference_no_tapering/raynest.h5', 'r') as f:
        no_EM_plpk_no_tapering_logZ= np.array(f['combined']['logZ'])
        post_noEM_plpk_no_tapering = np.array(f['combined']['posterior_samples'])

samples2 = np.column_stack([post_noEM_plpk_no_tapering[lab1] for lab1 in noEM_plpk_no_tapering_model.names])
fig = corner(samples2, labels = ['$M_1$', 'z_c'], truths = [85.5, None])
fig.savefig('inference_no_tapering/noEM_plpk_no_tapering_posterior.pdf', bbox_inches = 'tight')
