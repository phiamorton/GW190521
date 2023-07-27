

#overplot DL for conditioned likelihood dist, marginal likelihood dist, discovery paper IMR, DL(EM with Planck cosmology)

from os import read
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import h5py
from figaro.load import load_single_event
from GW190521_raynest import redshift_model
from figaro.load import load_density
import raynest
import raynest.model
from figaro.cosmology import CosmologicalParameters
from figaro.plot import plot_multidim , plot_1d_dist,plot_median_cr
from figaro.marginal import condition, marginalise
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from scipy.stats import gaussian_kde

#LVK new waveform
samples_in, name = load_single_event('GW190521.h5',par= ['luminosity_distance']) 
z_c=0.438
DL_em = CosmologicalParameters(0.674,0.315,0.685,-1.,0.).LuminosityDistance_double(z_c)
#fig=corner(samples_in[:], color='grey', label='LVK new waveform', plot_density=False, hist_kwargs ={'density':True, 'label':'LVK model old'})

D_L=np.linspace(1,9999,202)[1:-1]
#LVK old waveform
#dpgmm_file = 'old_waveform/draws_old_GW190521.pkl' #detector frame M_1 and DL
# draws = load_density(dpgmm_file)
# draws_pdf = np.mean([d.pdf(D_L.T) for d in draws], axis = 0).reshape(len(D_L)) 
#interp_old=CubicSpline(D_L, draws_pdf)    
#fig = plot_median_cr(draws, label = 'D_{effective}', unit='Mpc', median_label='LVK old waveform')
with h5py.File('old_waveform/GW190521_posterior_samples.h5', 'r') as f:
    post = np.array(f['IMRPhenomPv3HM']['posterior_samples'])
    #print(post['luminosity_distance'])
samples= np.column_stack(post['luminosity_distance'])
#print(samples_GW17)
kernel=gaussian_kde(samples)
plt.plot(D_L, kernel(D_L),linewidth=1, color='indigo', label='LVK v2:\nIMRPhenomPv3HM ')
#plt.hist()

#conditioned
dpgmm_file = 'conditioned_density_draws_M1_and_DL.pkl' #detector frame M_1 and DL
draws = load_density(dpgmm_file)
conditioned_draws=[d.marginalise([0]) for d in draws] 
#print([d.n_cl for d in conditioned_draws])
#plot_median_cr(conditioned_draws, fig=fig, samples = samples_in, true_value=DL_em, true_value_label= 'EM',  label = 'D_{effective}', unit='Mpc', median_label='conditional')
draws_pdf = np.mean([d.pdf(D_L.T) for d in conditioned_draws], axis = 0).reshape(len(D_L)) 
#interp_cond=interp1d(D_L, draws_pdf)    
#fig = plot_median_cr(draws, label = 'D_{effective}', unit='Mpc', median_label='LVK old waveform')
plt.plot(D_L,draws_pdf,linewidth=1, color='indianred', label='conditioned')
perc = np.sum(draws_pdf[D_L<DL_em]*(D_L[1]-D_L[0]))
print("The EM candidate lies at the", perc*100, "percentile")

# percentile=np.percentile(draws_pdf, [70])
# print(DL_em)
# #print(np.median(interp_cond(D_L))
# print(percentile)


#marginal
filepath= 'draws_allsky_GW190521/draws_GW190521.pkl' 
draws_marg = load_density(filepath)
draws_m1_marginal = marginalise(draws_marg, [0])#[d.marginalise([0]) for d in draws_marg] 
#plot_median_cr(draws_m1_marginal, fig = fig, median_label='marginal')
#corner(draws_m1_marginal, fig=fig)
draws_pdf = np.mean([d.pdf(D_L.T) for d in draws_m1_marginal], axis = 0).reshape(len(D_L)) 
#interp_marg=interp1d(D_L, draws_pdf)    
#fig = plot_median_cr(draws, label = 'D_{effective}', unit='Mpc', median_label='LVK old waveform')
plt.plot(D_L,draws_pdf, linewidth=1, color='forestgreen', label='marginalized')


#fig.axes.legend(*fig.axes.get_legend_handles_labels(), loc='center', frameon=False)
plt.axvline(DL_em, linewidth=1,color='mediumblue', label='EM counterpart')

plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
plt.ylim(bottom=0.)
plt.legend()
plt.savefig('Distance_comparison.pdf')
plt.savefig('DL_overplot.pdf')

