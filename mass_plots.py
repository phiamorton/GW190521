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
ax.plot(M, kernel(M), color='purple', label='Redshift model')


#LVK

with h5py.File('IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5', 'r') as f:
    post= np.array(f['C01:Mixed']['posterior_samples'])
    print(help(f['C01:Mixed']['posterior_samples']))







ax.legend()
ax.set_ylim(bottom=0.)
fig.savefig('mass_dist.pdf')
