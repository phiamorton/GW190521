import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.interpolate import interp1d

"""
LVK best estimates for the PL+Peak parameters
Data from Abbott+ (2022) https://arxiv.org/abs/2111.03634
Please note that Abbott+ does not provide an estimate for the peak width, we obtained it from the data release directly
"""
alpha_pl   = 3.5   #
mu_peak    = 34.   # Msun
sigma_peak = 4.6   # Msun
w          = 0.038 #
m_min      = 3.    # Msun

# Useful quantities
log_w     = np.log(w)
log_1mw   = np.log(1.-w)
norm_pl   = np.log(alpha_pl-1) - (1-alpha_pl)*np.log(m_min)
norm_peak = -0.5*np.log(2*np.pi) - np.log(sigma_peak)

# LVK interpolant
LVK_o3 = np.genfromtxt('lvk_log_plpeak.txt')
pl_peak_interpolant = interp1d(LVK_o3[:,0], LVK_o3[:,1], fill_value = 'extrapolate')

# H0 with GW170817
H0 = np.genfromtxt('H0_prior.txt')
H0_interpolant = interp1d(H0[:,0], H0[:,1], fill_value = 'extrapolate')

# AGN mass distribution (Paola)
m1_agn = np.genfromtxt('AGN_mass_distribution.txt')
agn_mass_interpolant = interp1d(m1_agn[:,0], m1_agn[:,1], fill_value = 'extrapolate')

def pl_peak_no_tapering(m):
    """
    Power-law + peak model without any tapering (low mass or high mass).
    """
    log_PL   = -alpha_pl*np.log(m) + norm_pl
    log_peak = -0.5*((m-mu_peak)/sigma_peak)**2 + norm_peak
    return np.logaddexp(log_1mw+log_PL, log_w+log_peak)

def pl_peak_LVK(m):
    """
    LVK Power-law + peak model as in Abbott et al (2022) https://arxiv.org/abs/2111.03634
    Data from https://zenodo.org/record/7843926
    """
    return pl_peak_interpolant(m)

def H0_prior(H0):
    """
    Prior induced by GW170817 EM counterpart
    """
    return H0_interpolant(H0)

def AGN_mass_prior(m):
    """
    AGN mass model from Paola's dataset
    """
    return agn_mass_interpolant(m)
    
def rad_prior(r):
    slope = 0.01 
    intercept=1

    #Laplacian/resonance peaks:
    center1=330
    center2=24.5   #https://arxiv.org/pdf/1511.00005.pdf 
    width1=5*np.log(2)
    width2=5*np.log(2) #2 R_s on log scale
    linear= slope * r +intercept
    #normalize it in log space??????
    #area= logslope*logr.max()**2+intercept*logr.max() - (logslope*logr.min()**2+intercept*logr.min())
    #linear/=area

    #c=2 the scipy documentation is a lie
    #defined with a center and scale in log space
    #just eyeballed scale factor to shrink the width and then rescaled the entire distribution 
    peak1= (laplace.pdf(r, center1, width1)) *50
    peak2=(laplace.pdf(r, center2, width2)) *50
    return linear + peak1 +peak2      # /3 if each is normalized IF WANT TO NORMALIZE? MAYBE?

if __name__ == '__main__':

    #radius (in terms of Swarzchild radii) on a log scale
    r= np.linspace(1, 400, 100000) #already log
    #print(logr)

    plt.plot(r, rad_prior(r))

    #print(help(laplace.ppf))
    #test1=laplace.pdf(logr,2,width1/100)
    #plt.plot(logr, test1)
    plt.xlabel(r'Distance from SMBH [${R_s}$]')
    plt.ylabel('prob(BBH location)')
    plt.savefig('radius_prior')

