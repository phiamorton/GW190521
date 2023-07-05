import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
from corner import corner

from scipy.special import logsumexp
from numba import njit

from figaro.cosmology import CosmologicalParameters
from figaro.load import load_density
from figaro.likelihood import logsumexp_jit

from radius_prior import rad_prior

class redshift_model(raynest.model.Model):

    def __init__(self, z_c, draws):
        super(redshift_model,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.draws=draws
        self.N_draws = len(self.draws)
        self.ones    = np.ones(self.N_draws)
        self.z_c=z_c
        self.omega = CosmologicalParameters(0.674, 0.315, 0.685, -1., 0.)
        self.DL_em = self.omega.LuminosityDistance_double(self.z_c)
        
        self.names= ['r', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_C', # M_C true chirp mass
                     'angle_disk_RA', # theta_disk is the inclination of the AGN disk (max at 0)
                     'angle_disk_DEC', # theta_disk is the inclination of the AGN disk (max at 0)
                     'orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at pi/2), axis defined as orthogonal to LOS 
                     ]

       # equations using G=c=1
       #orbital phases defined in the plane (ie along the disk from bird's eye view)
    #    'D_L', # D_L true luminosity distance in Mpc
    #                  'z_rel', # z_rel   is relativistic redshift
    #                  'z_grav', # z_grav gravitational redshift
    # 'D_eff', # D_eff is effective luminosity distance from GW data in Mpc
    #                  'M_eff', # M_eff is from GW data

       #need to use bounds in log space
        self.bounds =[ [0,10], [0,300], [0,np.pi], [0, 2* np.pi], [0, 2*np.pi] ]

    

    def log_prior(self,x):
        logp=super(redshift_model,self).log_prior(x)

        #code prior on r:
        #width on resonances ~1-2 R_s centered on torque distances from Peng+ 2021 0.85 and 1.62 approx on log scale
        #source: https://arxiv.org/pdf/2104.07685.pdf
        #use radius prior from radius_prior.py

        if np.isfinite(logp):
            logp_radius = 0.#np.log(rad_prior(x['r'])) #radius prior (log Swarzchild radii)
            logp_M_c    = 0. #agnostic flat chirp mass prior
            #could replace with LVK informed prior
            return logp_radius + logp_M_c
        else:
            return -np.inf

    def log_likelihood(self,x):
        
        #need prob(D_L eff and M_c eff from GW data), use distribution from FIGARO?
        #need to normalize this distribution with the prior on D_L eff and M_c eff ? not sure how to get this, somewhow from GW data

        #easiest to define velocity even though it is not a parameter directly used, but we need to relationship between r, v and z
        #the pre-merger velocity along LOS
        vel     = 1./np.sqrt(2*(np.exp(x['r'])-1))
        vel_LoS = vel * np.cos(x['angle_disk_RA']) * np.cos(x['angle_disk_DEC']) * np.cos(x['orbital_phase'])
        #gamma/lorentz factor
        gamma = 1./np.sqrt(1 - vel**2)

        #z_rel (r, angles)
        #uh oh need to check that I am looking at z_rel redshifted not blueshifted- need to be careful about angles
        z_rel = gamma * (1 - vel_LoS)

        #z_grav (r)
        z_grav = np.sqrt(1 - (x['r'])**-1)-1

        #D_L eff (z_c, z_rel, z_grav, D_L)
        D_eff = (1+z_rel)**2 * (1+z_grav) * self.DL_em

        #M_c eff (z_c, z_r, z_g, M_c)
        M_eff = (1+self.z_c) * (1+ z_rel) * (1+ z_grav) * x['M_c']

        #my likelihood is marginalized over D_L eff, M_c eff, angles, z_r, z_g, D_L
        logl = @njit(logsumexp_jit([d._fast_logpdf(np.atleast_2d([M_eff, D_eff])) for d in self.draws], self.ones) - np.log(self.N_draws))

        return logl

if __name__ == '__main__':

    dpgmm_file = 'path/to/file'
    z_c = 0.438
    GW_posteriors = load_density(dpgmm_file)

    mymodel= redshift_model(z_c, GW_posteriors)
    nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000)
    nest.run(corner = True)
    post = nest.posterior_samples.ravel()

    samples = np.column_stack([post[lab] for lab in mymodel.names])
    # fig = corner(samples, labels = ['$$','$$'])
    # fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')

