from figaro.load import load_density,  save_density
import numpy as np
import matplotlib.pyplot as plt
from figaro.plot import plot_multidim 
from figaro.marginal import condition

"""start with pkl file of r draws from figaro 
optputs a file with conditioned draws and a plot of the conditioned distribution"""


filepath= 'draws_GW190521.pkl'

ra_EM, dec_EM = 192.42625 , 34.8247

ra_EM_rad= ra_EM/360*np.pi*2
dec_EM_rad=dec_EM/180*np.pi 

draws = load_density(filepath)

conditioned_draws = condition(draws,[ra_EM_rad,dec_EM_rad], [2,3], norm=True, filter=True, tol=1e-3)
plot_multidim(conditioned_draws, name= 'conditioned_distribution', labels=['M_c', 'D_L'], units=['M_\\odot', 'Mpc'])

save_density(conditioned_draws, name='conditioned_density_draws')

print([d.n_cl for d in conditioned_draws])
