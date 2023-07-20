import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
from corner import corner
import h5py

from scipy.special import logsumexp
from numba import njit

from figaro.cosmology import CosmologicalParameters
from figaro.load import load_density
from figaro.likelihood import logsumexp_jit
from figaro.plot import plot_multidim 

from priors import rad_prior, pl_peak_LVK, pl_peak_no_tapering 


class noEM_model_plpk(raynest.model.Model):

    def __init__(self, draws):
        super(noEM_model_plpk,self).__init__()
        self.draws=draws
        self.N_draws = len(self.draws)
        self.ones    = np.ones(self.N_draws)
        self.omega   = CosmologicalParameters(0.674,0.315,0.685,-1.,0.)
        
        self.names= ['M_C',# M_C true chirp mass
                     'z_c'] #D_L luminosity distance
                
        self.bounds =[ [0.,300.], [0,2] ]
        
    def log_prior(self,x):

        logp=super(noEM_model_plpk,self).log_prior(x)

        if np.isfinite(logp):    
            logp_M_c = pl_peak_LVK(x['M_C']) #power law +peak ??
            logp_z   = np.log(self.omega.ComovingVolumeElement_double(x['z_c'])) #unifrom in comoving volume 
            return logp_M_c +logp_z
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL = self.omega.LuminosityDistance_double(x['z_c'])
        
        M_eff = (1+x['z_c'])* x['M_C'] #chirp mass with cosmological redshift= M_eff

        pt = np.atleast_2d([M_eff, DL]) #DL not conditioned on EM candidate
        
        logl = self.draws[0]._fast_logpdf(pt)  #one draw
        return logl


dpgmm_file = 'draws_GW190521_mc_dist.pkl' #detector frame M_C, not conditioned

GW_posteriors = load_density(dpgmm_file)

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
fig = corner(samples1, labels = ['$M_c$', 'z_c'], truths = [63.3, None])
fig.savefig('inference_noEM_plpk/noEM_plpk_posterior.pdf', bbox_inches = 'tight')


class noEM_model_plpk_no_tapering(raynest.model.Model):

    def __init__(self, draws):
        super(noEM_model_plpk_no_tapering,self).__init__()
        self.draws=draws
        self.N_draws = len(self.draws)
        self.ones    = np.ones(self.N_draws)
        self.omega   = CosmologicalParameters(0.674,0.315,0.685,-1.,0.)
        
        self.names= ['M_C',# M_C true chirp mass
                     'z_c'] #D_L luminosity distance
                
        self.bounds =[ [0.,300.], [0,2] ]
        
    def log_prior(self,x):

        logp=super(noEM_model_plpk,self).log_prior(x)

        if np.isfinite(logp):    
            logp_M_c = pl_peak_no_tapering(x['M_C']) #power law +peak  but without the tapering
            logp_z   = np.log(self.omega.ComovingVolumeElement_double(x['z_c'])) #unifrom in comoving volume 
            return logp_M_c +logp_z
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL = self.omega.LuminosityDistance_double(x['z_c'])
        
        M_eff = (1+x['z_c'])* x['M_C'] #chirp mass with cosmological redshift= M_eff

        pt = np.atleast_2d([M_eff, DL]) #DL not conditioned on EM candidate
        
        logl = self.draws[0]._fast_logpdf(pt)  #one draw
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

samples1 = np.column_stack([post_noEM_plpk[lab1] for lab1 in noEM_plpk_model.names])
fig = corner(samples1, labels = ['$M_c$', 'z_c'], truths = [63.3, None])
fig.savefig('inference_no_tapering/noEM_plpk_no_tapering_posterior.pdf', bbox_inches = 'tight')