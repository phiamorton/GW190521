from urllib.parse import non_hierarchical
from numpy.ma.core import nonzero
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

from priors import rad_prior, AGN_mass_prior

class redshift_model(raynest.model.Model):

    def __init__(self, z_c, draws):
        super(redshift_model,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.draws=draws
        # self.N_draws = len(self.draws)
        # self.ones    = np.ones(self.N_draws)
        self.z_c=z_c
        self.DL_em = CosmologicalParameters(0.674,0.315,0.685,-1.,0.).LuminosityDistance_double(self.z_c)
        
        self.names= ['r', # radius from SMBH in terms of Swarzshild radii 
                     'M_1', # M_1 true mass (NOT detector frame)
                     #'angle_disk_RA', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'angle_disk_DEC', # theta_disk is the inclination of the AGN disk (max at 0)
                     #'orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at 0), axis defined as orthogonal to LOS 
                     'cos_effective_angle'] #I dont really care about the relative angle, only need one effective angle between LoS and GW emission, sampled uniform in cos
                     #'H_0',#Hubble constant
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
        self.bounds =[ [3, 400], [0.,200.], [-1,1] ]
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
            #logp_radius = 0.
            logp_radius= np.log(rad_prior(x['r'])) #radius prior ( Swarzchild radii)
            #https://drive.google.com/drive/folders/1aCHNObF4t6usq2H18MGR9PcsgiJr-PyB 
            #do I need to normalize???
            #when I add this prior and make r/R_s bounds larger: theta_eff peaks at 0 without the little symmetric peaks? weird?
            logp_M = AGN_mass_prior(x['M_1']) #agnostic flat chirp mass prior 
            #mass dist. from Paola

            return logp_radius + logp_M
        else:
            return -np.inf

    def log_likelihood(self,x):
        
        #luminosity distance based on cosomology and z from EM counterpart
        
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
        D_eff = (1+z_rel)**2 * (1+z_grav) * self.DL_em

        #M_c eff (z_c, z_r, z_g, M_c)
        M_eff = (1+self.z_c) * (1 + z_rel) * (1 + z_grav) * x['M_1']

        #pt = np.atleast_2d([M_eff, D_eff])
        #my likelihood is marginalized over D_L eff, M_c eff, z_r, z_g, D_L
        #logl = self.draws[0]._fast_logpdf(pt)  #one draw
        #logl = logsumexp_jit(np.array([d._fast_logpdf(pt) for d in self.draws[:10]]), self.ones) - np.log(self.N_draws)  #average of multiple d
        logl=GW_post(M_eff,D_eff)
        #print(2*np.log(D_eff))
        logl -= 2*np.log(D_eff) #remove GW prior
        # logl = logsumexp_jit(np.array([d._fast_logpdf(pt) for d in self.draws]), self.ones) - np.log(self.N_draws)  #average of multiple draws

        return logl

if __name__ == '__main__':

    postprocess=True

    #dpgmm_file = 'primarymass/conditioned_density_draws_M1_and_DL.pkl' #non-redshifted M_1
    #the conditional distribution (based on EM sky location)
    #z_c from EM counterpart candidate https://arxiv.org/pdf/2006.14122.pdf at ~2500 Mpc
    #GW_posteriors = load_density(dpgmm_file)
    dpgmm_file= 'conditional_interpolation_nF.pkl'
    with open(dpgmm_file, 'rb') as f:
        GW_posteriors = pickle.load(f)
    def GW_post(M,DL):
        return GW_posteriors(M,DL)
    z_c = 0.438
    

    mymodel= redshift_model(z_c, GW_posteriors)
    if not postprocess:
        nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000, output = 'inference_M1_final/')
        nest.run(corner = True)
        post = nest.posterior_samples.ravel()
    else:
        with h5py.File('inference_M1_final/raynest.h5', 'r') as f:
            post = np.array(f['combined']['posterior_samples'])

    samples = np.column_stack([post[lab] for lab in mymodel.names])
    #samples[:,0] = np.exp(samples[:,0])
    fig = corner(samples, labels = ['Distance from SMBH $[R_s]$','$M_1 [M_\odot]$', '$cos(\\theta_{effective})$'], truths=[None,98.4, None], show_titles=True) #, truths = [None,None,None,67.4,0.315]) #'$RA$','$Dec$','$phase$'])
    #might be a good visual to add M_C unredshifted as reported by LVK to compare
   
    fig.savefig('inference_M1_final/joint_posterior_redshiftmodel_M1_interp.pdf', bbox_inches = 'tight')

    #now plotting a comparison of the figaro reconstruction versus the output for D_Leff and M_c
    # need to return the D_L_eff distribution from my model? 

    #here is a corner plot of only r and M_c
    omega = CosmologicalParameters(0.674, 0.315, 0.685, -1., 0.)
    DL_em = omega.LuminosityDistance_double(z_c)
    reconstruction= samples[:,[0,1]]
    #reconstruction[:,0]=np.exp(reconstruction[:,0]) #use if samples of r are log
    r=samples[:,0]
    vel=1./np.sqrt(2*(r-1))  #the magnitude at a given distance from SMBH
    vel_LoS = vel * samples[:,2] #* np.cos(samples[:,3]) * np.cos(samples[:,4]) #Ive created a monster :((
    #gamma=lorentz factor
    gamma = 1./np.sqrt(1 - vel**2)

    #z_rel (r, angles)
    #make bounds on angle 0 to 2pi, redshifted should be when v_LoS is negative (theta=pi)
    z_rel = gamma * (1 + vel_LoS) - 1

    #z_grav (r)
    z_grav = 1./np.sqrt(1 -(reconstruction[:,0])**-1 ) - 1 
    #D_L eff (z_c, z_rel, z_grav, D_L)
    D_eff = (1+z_rel)**2 * (1+z_grav) * DL_em 

    reconstruction[:,0]=D_eff

    M_eff = (1+z_c) * (1 + z_rel) * (1 + z_grav) * samples[:,1]

    reconstruction[:,1]=M_eff 

    percentile=np.percentile(samples[:,1], [5, 95])
    median=np.median(samples[:,1])
    print("primary mass credible interval" ,median, "-", median-percentile[0],"+", percentile[1]-median)

    #percentile=np.percentile(D_eff, [80])
    #print(DL_em)
    #print(percentile)
    #median=np.median(D_eff)
    #print(median, "-", median-percentile[0],"+", percentile[1]-median)


    fig2=corner(reconstruction[:,[1,0]],labels = [ 'M_1 effective ',' D_L effective'], truths=[98.4,None]) 
    fig2.savefig('inference_M1_final/GW_posterior_vs_reconstruction_redshift_M1_interp.pdf', bbox_inches = 'tight')
    
    