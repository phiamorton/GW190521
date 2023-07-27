import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gaussian_kde
from figaro import plot_settings

M=np.linspace(50,150, 10**3)
fig, ax = plt.subplots()

#redshift
with h5py.File('inference_M1_final/raynest.h5', 'r') as f:
    post = np.array(f['combined']['posterior_samples'])

samples = np.column_stack(post['M_1'] )
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M), linewidth=1.2, color='indianred', label='Redshift model')


#LVK
#h5 file from combined waveform https://gwosc.org/eventapi/html/GWTC-2.1-confident/GW190521/v4/
with h5py.File('IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5', 'r') as f:
    post= np.array(f['C01:Mixed']['posterior_samples']['mass_1_source'])
    #print(post)

samples = np.column_stack(post )
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M), linewidth=1.2, color='forestgreen', label='LVK v4 model')


with h5py.File('old_waveform/GW190521_posterior_samples.h5', 'r') as f:
    post = np.array(f['IMRPhenomPv3HM']['posterior_samples'])
    #print(post['luminosity_distance'])
samples= np.column_stack(post['mass_1_source'])
#print(samples_GW17)
kernel=gaussian_kde(samples)
ax.plot(M, kernel(M),linewidth=1.2, color='indigo', label='LVK v2:\nIMRPhenomPv3HM ')



ax.legend()
ax.set_ylim(bottom=0.)
ax.set_xlabel('Primary Mass [$M\\odot$]')
ax.set_ylabel('Probability')
fig.savefig('mass_dist.pdf')
