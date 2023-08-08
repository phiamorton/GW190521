from os import PRIO_PROCESS
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

from priors import rad_prior, H0_prior, AGN_mass_prior

class redshift_model(raynest.model.Model):

    def __init__(self, z_c, draws):
        super(redshift_model,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.draws=draws
        # self.N_draws = len(self.draws)
        # self.ones    = np.ones(self.N_draws)
        self.z_c=z_c
        
        
        self.names= ['r', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_1', # M_C true chirp mass
                     #'angle_disk_RA', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'angle_disk_DEC', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at 0), axis defined as orthogonal to LOS 
                     'cos_effective_angle', #I dont really care about the relative angle, only need one effective angle between LoS and GW emission, sampled uniform in cos
                     'H_0']#,#Hubble constant
                     #'om'] #matter density

    # equations using G=c=1
    #orbital phases defined in the plane (ie along the disk from bird's eye view)
    #  'D_L', # D_L true luminosity distance in Mpc
    # 'z_rel', # z_rel   is relativistic redshift
    #  'z_grav', # z_grav gravitational redshift
    # 'D_eff', # D_eff is effective luminosity distance from GW data in Mpc
    # 'M_eff', # M_eff is from GW data

       #need to use bounds in log space
       #ISCO at 3R_S https://en.wikipedia.org/wiki/Innermost_stable_circular_orbit 
        self.bounds =[ [3,400], [0.,200.], [-1,1],[10,200]] 
        #[0,2*np.pi], [0, 2*np.pi], [0, 2*np.pi] ] #for using 3 angles
        #updated r bounds to show whole region where migration traps may occurr and 
        # have minimum at ISCO, no longer gives invalid sqrt error
        #LVK reported chirp mass (63.3 +19.6 -14.6) M_sun https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190521/v4/ 

    

    def log_prior(self,x):
        logp=super(redshift_model,self).log_prior(x)

        #code prior on r:
        #width on resonances ~1-2 R_s centered on torque distances from Peng+ 2021 0.85 and 1.62 approx on log scale
        #source: https://arxiv.org/pdf/2104.07685.pdf
        #use radius prior from radius_prior.py


        if np.isfinite(logp):
    
            logp_radius= np.log(rad_prior(x['r'])) #radius prior ( Swarzchild radii)
            logp_M_c = AGN_mass_prior(x['M_1'])

            return logp_radius + logp_M_c
        else:
            return -np.inf

    def log_likelihood(self,x):
        #with Hubble constant and matterdensity, z from EM galaxy redshift, find D_L cosmological
        #omega = CosmologicalParameters(x['H_0'], x['om'], 1-x['om'], -1., 0.)
        #luminosity distance based on cosomology and z from EM counterpart
        DL_em = CosmologicalParameters(x['H_0']/100, 0.315, 0.685, -1., 0.).LuminosityDistance_double(self.z_c)
        #remember to normalize H_0
        #DL_em=omega.LuminosityDistance_double(self.z_c)
        #need prob(D_L eff and M_c eff from GW data), use distribution from FIGARO?

        #easiest to define velocity even though it is not a parameter directly used, but we need to relationship between r, v and z
        #the pre-merger velocity along LOS
        vel = 1./np.sqrt(2*(x['r']-1))
        vel_LoS = vel * (x['cos_effective_angle']) #np.cos(x['angle_disk_RA']) * np.cos(x['angle_disk_DEC']) * np.cos(x['orbital_phase'])
        #gamma/lorentz factor
        gamma = 1./np.sqrt(1 - vel**2)

        #z_rel (r, angles)
        #uh oh am  I looking at z_rel redshifted not blueshifted?- need to be careful about angles
        #peak at theta=pi ==blueshift
        #in Alejandros -vel_LoS but on wiki +vel_LoS (https://en.wikipedia.org/wiki/Relativistic_Doppler_effect)
        z_rel = gamma * (1 + vel_LoS) - 1     #is it + or - ???? I do not know??? https://physics.stackexchange.com/questions/61946/relativistic-doppler-effect-derivation 

        #z_grav (r)
        z_grav = 1./np.sqrt(1 - 1./x['r']) - 1 
        #D_L eff (z_c, z_rel, z_grav, D_L)
        D_eff = (1+z_rel)**2 * (1+z_grav) * DL_em

        #M_c eff (z_c, z_r, z_g, M_c)
        M_eff = (1+self.z_c) * (1 + z_rel) * (1 + z_grav) * x['M_1']

        #pt = np.atleast_2d([M_eff, D_eff])
        #my likelihood is marginalized over D_L eff, M_c eff, z_r, z_g, D_L
        #logl = self.draws[0]._fast_logpdf(pt)  #one draw
        logl=GW_post(M_eff,D_eff)
        logl-=2*np.log(D_eff) #remove GW prior
        # logl = logsumexp_jit(np.array([d._fast_logpdf(pt) for d in self.draws]), self.ones) - np.log(self.N_draws)  #average of multiple draws

        return logl



class redshift_model_GW17(raynest.model.Model):

    def __init__(self, z_c, draws):
        super(redshift_model_GW17,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.draws=draws
        # self.N_draws = len(self.draws)
        # self.ones    = np.ones(self.N_draws)
        self.z_c=z_c
        
        
        self.names= ['r', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_1', # M_C true chirp mass
                     #'angle_disk_RA', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'angle_disk_DEC', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at 0), axis defined as orthogonal to LOS 
                     'cos_effective_angle', #I dont really care about the relative angle, only need one effective angle between LoS and GW emission, sampled uniform in cos
                     'H_0']#,#Hubble constant
        self.bounds =[ [3,400], [0.,200.], [-1,1],[10,200]] 
        
    def log_prior(self,x):
        logp=super(redshift_model_GW17,self).log_prior(x)

        if np.isfinite(logp):
     
            logp_radius= np.log(rad_prior(x['r'])) #radius prior ( Swarzchild radii)
            logp_M = AGN_mass_prior(x['M_1'])
            logp_H= H0_prior(x['H_0'])
           
            return logp_radius + logp_M +logp_H
        else:
            return -np.inf

    def log_likelihood(self,x):
        DL_em = CosmologicalParameters(x['H_0']/100, 0.315, 0.685, -1., 0.).LuminosityDistance_double(self.z_c)
        vel = 1./np.sqrt(2*(x['r']-1))
        vel_LoS = vel * (x['cos_effective_angle']) #np.cos(x['angle_disk_RA']) * np.cos(x['angle_disk_DEC']) * np.cos(x['orbital_phase'])
        
        gamma = 1./np.sqrt(1 - vel**2)
        z_rel = gamma * (1 + vel_LoS) - 1     
        
        z_grav = 1./np.sqrt(1 - 1./x['r']) - 1 
        
        D_eff = (1+z_rel)**2 * (1+z_grav) * DL_em

        M_eff = (1+self.z_c) * (1 + z_rel) * (1 + z_grav) * x['M_1']
        
        logl=GW_post(M_eff,D_eff)
        logl-=2*np.log(D_eff) #remove GW prior
        
        return logl

if __name__ == '__main__':

    postprocess=True

    dpgmm_file= 'conditional_interpolation_nF.pkl'
    with open(dpgmm_file, 'rb') as f:
        GW_posteriors = pickle.load(f)
    def GW_post(M,DL):
        return GW_posteriors(M,DL) 
    
    z_c = 0.438
    GW_posteriors = load_density(dpgmm_file)

    mymodel= redshift_model(z_c, GW_posteriors)
    if not postprocess:
        nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference_H/')
        nest.run(corner = True)
        post = nest.posterior_samples.ravel()
    else:
        with h5py.File('inference_H/raynest.h5', 'r') as f:
            post = np.array(f['combined']['posterior_samples'])

    samples = np.column_stack([post[lab] for lab in mymodel.names])
    # samples[:,0] = np.exp(samples[:,0])
    fig = corner(samples, labels = ['$\\frac{r}{R_s}$','$M_1 [M_\\odot]$', '$cos(\\theta)$', '$H_0$'], truths = [None,None,None,67.4], titles= ['$\\frac{r}{R_s}$','$M_1$', '$cos(\\theta)$', '$H_0$'], show_titles=True) #'$RA$','$Dec$','$phase$'])
    #might be a good visual to add M_C unredshifted as reported by LVK to compare
    fig.savefig('inference_H/joint_posterior_with_H0.pdf', bbox_inches = 'tight')


    mymodel= redshift_model_GW17(z_c, GW_posteriors)
    if not postprocess:
        nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference_H_GW17/')
        nest.run(corner = True)
        post = nest.posterior_samples.ravel()
    else:
        with h5py.File('inference_H_GW17/raynest.h5', 'r') as f:
            post = np.array(f['combined']['posterior_samples'])

    samples_GW17 = np.column_stack([post[lab] for lab in mymodel.names])
    #samples_GW17[:,0] = np.exp(samples[:,0])
    fig2 = corner(samples_GW17, labels = ['$\\frac{r}{R_s}$','$M_1 [M_\\odot]$', '$cos(\\theta)$', '$H_0$'], truths = [None,None,None,67.4], titles= ['$\\frac{r}{R_s}$','$M_1$', '$cos(\\theta)$', '$H_0$'], show_titles=True) #'$RA$','$Dec$','$phase$'])
    #might be a good visual to add M_C unredshifted as reported by LVK to compare
    fig2.savefig('inference_H_GW17/joint_posterior_with_H0.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots()
ax.hist(samples_GW17[:,[3]], histtype='step', density = True)
ax.hist(samples[:,[3]], histtype='step', density=True)
ax.axvline(67.4, label='Planck')
ax.set_xlabel('$H_0$ estimate')
fig.savefig('H_0_estimate.pdf')