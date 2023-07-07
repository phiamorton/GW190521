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
        #print(self.DL_em)
        self.names= ['r', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_C', # M_C true chirp mass
                     'angle_disk_RA', # theta_disk is the inclination of the AGN disk (max at 0)
                     'angle_disk_DEC', # theta_disk is the inclination of the AGN disk (max at 0)
                     'orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at 0), axis defined as orthogonal to LOS 
                     ]

    # equations using G=c=1
    #orbital phases defined in the plane (ie along the disk from bird's eye view)
    #  'D_L', # D_L true luminosity distance in Mpc
    # 'z_rel', # z_rel   is relativistic redshift
    #  'z_grav', # z_grav gravitational redshift
    # 'D_eff', # D_eff is effective luminosity distance from GW data in Mpc
    # 'M_eff', # M_eff is from GW data

       #need to use bounds in log space
       #ISCO at 3R_S
        self.bounds =[ [0, 3], [0.,300.], [0,2*np.pi], [0, 2*np.pi], [0, 2*np.pi] ]

    

    def log_prior(self,x):
        logp=super(redshift_model,self).log_prior(x)

        #code prior on r:
        #width on resonances ~1-2 R_s centered on torque distances from Peng+ 2021 0.85 and 1.62 approx on log scale
        #source: https://arxiv.org/pdf/2104.07685.pdf
        #use radius prior from radius_prior.py


        if np.isfinite(logp):
            logp_radius = 0.
            #logp_radius= np.log(rad_prior(x['r'])) #radius prior (log Swarzchild radii)
            logp_M_c = 0. #agnostic flat chirp mass prior
            #could replace with LVK informed prior
            return logp_radius + logp_M_c
        else:
            return -np.inf

    def log_likelihood(self,x):
        
        #need prob(D_L eff and M_c eff from GW data), use distribution from FIGARO?

        #easiest to define velocity even though it is not a parameter directly used, but we need to relationship between r, v and z
        #the pre-merger velocity along LOS
        vel = 1./np.sqrt(2*(np.exp(x['r'])-1))
        vel_LoS = vel * np.cos(x['angle_disk_RA']) * np.cos(x['angle_disk_DEC']) * np.cos(x['orbital_phase'])
        #gamma/lorentz factor
        gamma = 1./np.sqrt(1 - vel**2)

        #z_rel (r, angles)
        #uh oh need to check that I am looking at z_rel redshifted not blueshifted- need to be careful about angles
        z_rel = gamma * (1 + vel_LoS) - 1

        #z_grav (r)
        z_grav = 1./np.sqrt(1 - np.exp(-x['r'])) - 1 
        #D_L eff (z_c, z_rel, z_grav, D_L)
        D_eff = (1+z_rel)**2 * (1+z_grav) * self.DL_em

        #M_c eff (z_c, z_r, z_g, M_c)
        M_eff = (1+self.z_c) * (1 + z_rel) * (1 + z_grav) * x['M_C']

        pt = np.atleast_2d([M_eff, D_eff])
        #my likelihood is marginalized over D_L eff, M_c eff, z_r, z_g, D_L
        logl = self.draws[0]._fast_logpdf(pt)  #one draw
        # logl = logsumexp_jit(np.array([d._fast_logpdf(pt) for d in self.draws]), self.ones) - np.log(self.N_draws)  #average of multiple draws

        return logl

if __name__ == '__main__':

    postprocess = True

    dpgmm_file = 'conditioned_density_draws.pkl'
    #the conditional distribution (based on EM sky location)
    #z_c from EM counterpart candidate
    z_c = 0.438
    GW_posteriors = load_density(dpgmm_file)

    mymodel= redshift_model(z_c, GW_posteriors)
    if not postprocess:
        nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference/')
        nest.run(corner = True)
        post = nest.posterior_samples.ravel()
    else:
        with h5py.File('inference/raynest.h5', 'r') as f:
            post = np.array(f['combined']['posterior_samples'])

    samples = np.column_stack([post[lab] for lab in mymodel.names])
    samples[:,0] = np.exp(samples[:,0])
    fig = corner(samples, labels = ['$r/r_s$','$M_c$','$RA$','$Dec$','$phase$'])
    fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')

    #now plotting a comparison of the figaro reconstruction versus the output for D_Leff and M_c
    # need to return the D_L_eff distribution from my model? 

    #here is a corner plot of only r and M_c
    omega = CosmologicalParameters(0.674, 0.315, 0.685, -1., 0.)
    DL_em = omega.LuminosityDistance_double(z_c)
    reconstruction= samples[:,[0,1]]
    #reconstruction[:,0]=np.exp(reconstruction[:,0]) #use if samples of r are log
    r=samples[:,0]
    vel=1./np.sqrt(2*((r)-1))
    vel_LoS = vel * np.cos(samples[:,2]) * np.cos(samples[:,3]) * np.cos(samples[:,4]) #Ive created a monster :((
        #gamma/lorentz factor
    gamma = 1./np.sqrt(1 - vel**2)

        #z_rel (r, angles)
        #uh oh need to check that I am looking at z_rel redshifted not blueshifted- need to be careful about angles
    z_rel = gamma * (1 + vel_LoS) - 1

        #z_grav (r)
    z_grav = 1./np.sqrt(1 -(reconstruction[:,0])**-1 ) - 1 
        #D_L eff (z_c, z_rel, z_grav, D_L)
    D_eff = (1+z_rel)**2 * (1+z_grav) * DL_em 

    reconstruction[:,0]=D_eff

    M_eff = (1+z_c) * (1 + z_rel) * (1 + z_grav) * reconstruction[:,1]

    reconstruction[:,1]=M_eff 
    
    fig2=plot_multidim(GW_posteriors, samples = reconstruction[:,[1,0]],labels = [ 'M_c','D_L']) 
    fig2.savefig('GW_posterior_vs_reconstruction.pdf', bbox_inches = 'tight')
    plt.show()
    #print(r.shape, vel.shape, vel_LoS.shape)
    plt.bar(r,vel)
    plt.xlabel('distance from SMBH [$R_s$]')
    plt.ylabel('velocity [% c]')
    plt.show()

    plt.bar(r,z_rel)
    plt.xlabel('distance from SMBH [$R_s$]')
    plt.ylabel('$z_{rel}$')
    plt.show()

    plt.bar(r,z_grav)
    plt.xlabel('distance from SMBH [$R_s$]')
    plt.ylabel('$z_{grav}$')
    plt.show()
    

    