from os import name
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator

from scipy.stats import multivariate_normal as mn

from corner import corner


# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# xx,yy=np.meshgrid(x,y, indexing='ij')
# gauss = mn(mean = [1,1], cov = [[1,0],[0,1]]).pdf
# datag = np.array([gauss([xi,yi]) for xi,yi in zip(xx.flatten(),yy.flatten())]).reshape(11,22)
# interp = RegularGridInterpolator((x, y), datag)

# for xi,yi in zip(xx.flatten(),yy.flatten()):
#     diff =(interp([xi,yi])- gauss([xi,yi]) )
#     print(diff)

dpgmm_file = 'primarymass/marginalized_density_draws_M1_and_DL.pkl' #detector frame M_1 and DL, marginalized over sky position

from figaro.load import load_density, save_density
GW_posteriors = load_density(dpgmm_file)

draws = load_density(dpgmm_file)

#bounds over which to interpolate for the parameters
M_1=np.linspace(0,300,202)[1:-1]
D_L=np.linspace(0,10000,202)[1:-1]
#print(M_1.shape, D_L.shape)
MM, DD = np.meshgrid(M_1, D_L)

draws_pdf = np.log(np.mean([d.pdf(np.array([MM.flatten(), DD.flatten()]).T) for d in draws], axis = 0).reshape(len(M_1), len(D_L)) )

interp_figaro= RegularGridInterpolator((M_1, D_L), draws_pdf.T, bounds_error=False)


# print(interp_figaro([100,5000]))
# print(np.mean([d.pdf([100,5000]) for d in draws], axis = 0))
# print(interp_figaro([80,4000]))
# print(np.mean([d.pdf([80,4000]) for d in draws], axis = 0))

plt.contourf(DD, MM, interp_figaro(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1),len(D_L)))
# corner(M_1, D_L, draws[0].pdf(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1), len(D_L)) )
plt.savefig("checking_interpolant.pdf")

import pickle
filename='Marginalized_interpolation.pkl'
with open(filename, 'wb') as f:
    pickle.dump(interp_figaro, f)

#with open(dpgmm_file, 'rb') as f:
#     GW_posteriors= pickle.load(f)