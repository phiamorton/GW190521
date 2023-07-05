import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
#print('raynest')
from corner import corner


class redshift_model(raynest.model.Model):

    def __init__(self,data, z_c):
        super(redshift_model,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.data=data
        self.z_c=z_c

        
        self.names= ['r_from_SMBH', # radius from SMBH in terms of Swarzshild radii (log scale)?
                     'M_C', # M_C true chirp mass
                     'D_L', # D_L true luminosity distance in Mpc
                     'z_rel', # z_rel   is relativistic redshift
                     'z_grav', # z_grav gravitational redshift
                     'theta_disk', # theta_disk is the inclination of the AGN disk (max at 0)
                     'theta_orbital_phase', # theta_orbital_phase is the phase of BBH in its orbit (max at pi/2), axis defined as orthogonal to LOS 
                     'D_eff', # D_eff is effective luminosity distance from GW data in Mpc
                     'M_eff', # M_eff is from GW data
                     ]

       # equations using G=c=1

       #need to use bounds in log space
        self.bounds =[ [0,5], [0,150], [1000,10000], [0,2], [0,2], [0,np.pi/4], [np.pi/4, 3*np.pi/4], [1000, 10000], [0,150] ]

    

    def log_prior(self,x):
        logp=super(redshift_model,self).log_prior(x)

        #code prior on r:
        #width on resonances ~1-2 R_s centered on torque distances from Peng+ 2021 0.85 and 1.62 approx on log scale
        #source: https://arxiv.org/pdf/2104.07685.pdf
        #use radius prior from radius_prior.py

        if np.isfinite(logp):
            from radius_prior import rad_prior
            prior_on_radius= rad_prior() #radius prior (log Swarzchild radii)
            M_c_prior= 1 #agnostic flatt chirp mass prior
            #could replace with LVK informed prior
            total_prior=prior_on_radius + M_c_prior
            return total_prior
        else:
            return -np.inf

    def log_likelihood(self,x):
        
        #need prob(D_L eff and M_c eff from GW data), use distribution from FIGARO?
        #need to normalize this distribution with the prior on D_L eff and M_c eff ? not sure how to get this, somewhow from GW data

        #easiest to define velocity even though it is not a parameter directly used, but we need to relationship between r, v and z
        vel= 1/2 *np.log(2(x['r']-1))


        #z_rel (r, angles)

        #z_grav (r)

        #D_L (z_c (known), cosmology (chosen))

        #D_L eff (z_c, z_r, z_g, D_L)

        #M_c eff (z_c, z_r, z_g, M_c)

        #my likelihood is marginalized over D_L eff, M_c eff, angles, z_r, z_g, D_L
        
        return logl




mymodel= redshift_model()
nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000)
nest.run(corner = True)
post = nest.posterior_samples.ravel()

samples = np.column_stack([post[lab] for lab in mymodel.names])
# fig = corner(samples, labels = ['$$','$$'], truths=[a_val,b_val])
# fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')

