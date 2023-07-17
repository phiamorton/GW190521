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

from radius_prior import rad_prior

#model with redshift:
from GW190521_raynest import redshift_model


class noEM_model(raynest.model.Model):

    def __init__(self, draws):
        super(noEM_model,self).__init__()
        self.draws=draws
        self.N_draws = len(self.draws)
        self.ones    = np.ones(self.N_draws)
    
        
        
        self.names= [#'r', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_C',# M_C true chirp mass
                     'z_c'] #D_L luminosity distance
                
                     #'cos_effective_angle', #I dont really care about the relative angle, only need one effective angle between LoS and GW emission, sampled uniform in cos
                     #'H_0',#Hubble constant
                     #'om'] #matter density

    # equations using G=c=1

       
        self.bounds =[ [0.,200.], [0,2] ]
        
    def log_prior(self,x):
        logp=super(noEM_model,self).log_prior(x)

        #code prior on r:
        #width on resonances ~1-2 R_s centered on torque distances from Peng+ 2021 0.85 and 1.62 approx on log scale
        #source: https://arxiv.org/pdf/2104.07685.pdf
        #use radius prior from radius_prior.py


        if np.isfinite(logp):
            
            logp_M_c = 0 #power law +peak ??

            return logp_M_c
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL = CosmologicalParameters(0.674,0.315,0.685,-1.,0.).LuminosityDistance_double(x['z_c'])
        
        M_eff = (1+x['z_c'])* x['M_C'] #chirp mass with cosmological redshift

        pt = np.atleast_2d([M_eff, DL])
        
        logl = self.draws[0]._fast_logpdf(pt)  #one draw
        #logl-=2*np.log(D_eff) #remove GW prior
        # logl = logsumexp_jit(np.array([d._fast_logpdf(pt) for d in self.draws]), self.ones) - np.log(self.N_draws)  #average of multiple draws

        return logl

dpgmm_file = 'draws_GW190521_mc_dist.pkl' #non-redshifted M_c, nonconditioned

GW_posteriors = load_density(dpgmm_file)

postprocess=False

notmymodel= noEM_model(GW_posteriors)

if not postprocess:
    nest1 = raynest.raynest(notmymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference/')
    nest1.run(corner = True)
    post1 = nest1.posterior_samples.ravel()
else:
    with h5py.File('inference/raynest.h5', 'r') as f:
        post1 = np.array(f['combined']['posterior_samples'])

samples1 = np.column_stack([post1[lab] for lab in notmymodel.names])

z_EM = 0.438
mymodel= redshift_model(z_EM, GW_posteriors)
if not postprocess:
    nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference/')
    nest.run(corner = True)
    post = nest.posterior_samples.ravel()
else:
    with h5py.File('inference/raynest.h5', 'r') as f:
        post = np.array(f['combined']['posterior_samples'])

samples = np.column_stack([post[lab] for lab in mymodel.names])

fig = corner(samples1, labels = ['$M_c$', 'z_c'], truths = [63.3])
fig.savefig('noEM_posterior.pdf', bbox_inches = 'tight')

print("estimated logZ for no redshift = {0} ".format(nest1.logZ))
print("estimated logZ for redshift = {0} ".format(nest.logZ))
print("Log Bayes' Factor redshift model vs nonredshifted= ",nest.logZ/nest1.logZ) 



