import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gaussian_kde
from figaro import plot_settings
import seaborn as sns
sns.color_palette('colorblind')

M=np.linspace(55,140, 10**3)
fig, ax = plt.subplots()

#LVK old
with h5py.File('old_waveform/GW190521_posterior_samples.h5', 'r') as f:
    post = np.array(f['IMRPhenomPv3HM']['posterior_samples'])
    #print(post['luminosity_distance'])
samples= np.column_stack(post['mass_1_source'])
#print(samples_GW17)
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M),linewidth=1.2, ls='--', color = 'steelblue', label='Discovery paper')



#redshift
with h5py.File('inference_M1_final/raynest.h5', 'r') as f:
    post = np.array(f['combined']['posterior_samples'])

samples = np.column_stack(post['M_1'] )
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M), linewidth=1.2, color='#cc78bc', label='Association')

#LVK
#h5 file from combined waveform https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190521/v4/
with h5py.File('IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5', 'r') as f:
    post= np.array(f['C01:Mixed']['posterior_samples']['mass_1_source'])
    #print(post)

samples = np.column_stack(post )
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M), linewidth=1.2, ls='-.', c= 'lightsalmon', label='GWTC-2.1 (marginal)')



ax.legend()
ax.set_ylim(bottom=0.)
ax.set_xlim(58, 140)
ax.set_xlabel('$M_1 \ [M_\\odot]$')
ax.set_ylabel('$p(M_1)$')
fig.savefig('mass_dist.pdf')
