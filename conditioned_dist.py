from figaro.load import load_density,  save_density
import numpy as np
import matplotlib.pyplot as plt
from figaro.plot import plot_multidim
from figaro.marginal import condition

"""start with pkl file of r draws from figaro 
optputs a file with conditioned draws and a plot of the conditioned distribution"""

filepath= 'primarymass/draws_GW190521.pkl' #MUST be the one with primary mass (in primary mass directory)

#the conditioned dist. 
ra_EM, dec_EM = 192.42625 , 34.8247

ra_EM_rad= ra_EM/360*np.pi*2
dec_EM_rad=dec_EM/180*np.pi 

draws = load_density(filepath)

conditioned_draws = condition(draws,[ra_EM_rad,dec_EM_rad], [2,3], norm=True, filter=False, tol=1e-3)

#plot_multidim(conditioned_draws, name= 'conditioned_distribution_M1', labels=['M_1', 'D_L'], units=['M_\\odot', 'Mpc'])

save_density(conditioned_draws, name='conditioned_density_draws_M1_and_DL')

conditioned_draws_nF = condition(draws,[ra_EM_rad,dec_EM_rad], [2,3], norm=False, filter=False, tol=1e-3)
#print(conditioned_draws_nF)
save_density(conditioned_draws_nF, name='conditioned_density_draws_M1_and_DL_nF')

#dont want to use normalized draws because it fucks up the Bayes factors
#print([d.n_cl for d in conditioned_draws_nF])

#now the marginal one too from the no association model
make_marginal=False
if make_marginal:
    filepath= 'draws_allsky_GW190521/draws_GW190521.pkl' 
    draws = load_density(filepath)
    draws_m1_marginal = [d.marginalise([2,3]) for d in draws] 
    save_density(draws_m1_marginal, name='marginalized_density_draws_M1_and_DL')

