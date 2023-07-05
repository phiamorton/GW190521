import raynest
import raynest.model
import numpy as np
import matplotlib.pyplot as plt 
#print('raynest')
from corner import corner


class Line(raynest.model.Model):

    def __init__(self,data, z_c):
        super(Line,self).__init__()
        #defining effective luminosity and chirp mass, will get this from data??
        # z_c comes from the EM candidate
        self.data=data
        self.z_c=z_c

        
        self.names= ['r_from_SMBH', 'M_C', 'D_L', 'z_rel', 'z_grav', 'theta_disk', 'theta_orbital_phase', 'D_eff', 'M_eff']

       #radius from SMBH in terms of Swarzshild radii (~400 to 1500)?
       # M_C true chirp mass
       # D_L true luminosity distance in Mpc
       # z_rel   is relativistic redshift
       # z_grav gravitational redshift
       # theta_disk is the inclination of the AGN disk (max at 0)
       # theta_orbital_phase is the phase of BBH in its orbit (max at pi/2), axis defined as orthogonal to LOS 
       #D_eff is effective luminosity distance from GW data in Mpc
       #M_eff is from GW data
       #equations using G=c=1
        self.bounds =[ [400,1500], [0,150], [1000,10000], [0,1], [0,1], [0,np.pi/4], [np.pi/4, 3*np.pi/4], [1000, 10000], [0,150] ]

    

    def log_prior(self,x):
        logp=super(Line,self).log_prior(x)
        if np.isfinite(logp):
            #some prior
            return 0.
        else:
            return -np.inf

    def log_likelihood(self,x):
        
        
        return logl




mymodel= Line()
nest = raynest.raynest(mymodel, verbose=2, nnest=1, nensemble=1, nlive=1000, maxmcmc=5000)
nest.run(corner = True)
post = nest.posterior_samples.ravel()

samples = np.column_stack([post[lab] for lab in mymodel.names])
# fig = corner(samples, labels = ['$$','$$'], truths=[a_val,b_val])
# fig.savefig('joint_posterior.pdf', bbox_inches = 'tight')

