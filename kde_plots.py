import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from figaro import plot_settings
import h5py
import seaborn as sns
sns.color_palette('dark')


data= np.genfromtxt('H_0.csv') #https://dcc.ligo.org/LIGO-P1700296/public

kernel=gaussian_kde(data)

fig, ax = plt.subplots()
#H_0 from H_0 paper from Planck and ShoES measurement med +-1 sigma

ax.axvspan( 67.4-0.5, 67.4+0.5, alpha=0.6, label='Planck', color='skyblue' ) #67.4 +-0.5 from 2018 https://arxiv.org/abs/1807.06209 
ax.axvspan( 67.4-1, 67.4+1, alpha=0.5, color='skyblue') #67.4 +-0.5 from 2018 https://arxiv.org/abs/1807.06209 
# ax.axvline(67.4, lw=1, color='skyblue')
# ax.axvline( 73.04, lw=1, color ='wheat')
ax.axvspan( 72.04, 74.06, alpha=0.6, label='SH0ES', color='wheat') #https://arxiv.org/abs/2112.04510 73.04+-1.04
ax.axvspan( 71.04, 75.1,  alpha=0.5, color='wheat') #https://arxiv.org/pdf/2112.04510.pdf 

H0=np.linspace(20,200, 10**3)
ax.plot(H0, kernel(H0), linewidth=1, ls='-.', c='indigo', label='GW170817')

#with prior
with h5py.File('inference_H_GW17/raynest.h5', 'r') as f:
    post = np.array(f['combined']['posterior_samples'])
    samples_GW17 = np.column_stack(post['H_0'])
#print(samples_GW17)
kernel=gaussian_kde(samples_GW17)
ax.plot(H0, kernel(H0), linewidth=1, c='darkorchid', label='GW190521 + GW170817')



#without prior
with h5py.File('inference_H/raynest.h5', 'r') as f:
    post = np.array(f['combined']['posterior_samples'])
    samples_H= np.column_stack(post['H_0'])
#print(samples_H)
kernel=gaussian_kde(samples_H)
ax.plot(H0, kernel(H0), linewidth=1, ls='--', c='forestgreen', label='GW190521')



ax.set_xlabel('$H_0\ [\\mathrm{km} \ \\mathrm{s}^{-1} \ \\mathrm{Mpc}^{-1}]$')
ax.set_ylabel('p($H_0$)')
ax.legend(loc=0)
ax.set_xlim(50,150)#H0.min(), H0.max())
ax.set_ylim(bottom = 0.)
fig.savefig('H_0_kdeplot.pdf', bbox_extra_artists=(), bbox_inches='tight')

#set lower limit 0
#https://dcc.ligo.org/public/0145/P1700296/005/LIGO-P1700296.pdf

