from os import name
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator, interp2d

#from scipy.stats import multivariate_normal as mn

from corner import corner
from figaro.load import load_density, save_density

# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# xx,yy=np.meshgrid(x,y, indexing='ij')
# gauss = mn(mean = [1,1], cov = [[1,0],[0,1]]).pdf
# datag = np.array([gauss([xi,yi]) for xi,yi in zip(xx.flatten(),yy.flatten())]).reshape(11,22)
# interp = RegularGridInterpolator((x, y), datag)

# for xi,yi in zip(xx.flatten(),yy.flatten()):
#     diff =(interp([xi,yi])- gauss([xi,yi]) )
#     print(diff)

marginal_density=True
if marginal_density:
    dpgmm_file = 'marginalized_density_draws_M1_and_DL.pkl' #detector frame M_1 and DL, marginalized over sky position
    #from all sky density, not just northern hemishpere


    draws = load_density(dpgmm_file)

    #bounds over which to interpolate for the parameters
    M_1=np.linspace(0,300,202)[1:-1]
    D_L=np.linspace(0,10000,202)[1:-1]
    #print(M_1.shape, D_L.shape)
    MM, DD = np.meshgrid(M_1, D_L)

    draws_pdf = np.log(np.mean([d.pdf(np.array([MM.flatten(), DD.flatten()]).T) for d in draws], axis = 0).reshape(len(D_L), len(M_1)) )
    print([d.pdf([100,2500]) for d in draws])
    interp_figaro_m= interp2d(M_1, D_L, draws_pdf, bounds_error=False)
    print("marginal", interp_figaro_m(100,2500))
    # print(interp_figaro([100,5000]))
    # print(np.mean([d.pdf([100,5000]) for d in draws], axis = 0))
    # print(interp_figaro([80,4000]))
    # print(np.mean([d.pdf([80,4000]) for d in draws], axis = 0))

    plt.contourf( DD, MM, interp_figaro_m(M_1, D_L))
    # corner(M_1, D_L, draws[0].pdf(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1), len(D_L)) )
    plt.savefig("checking_interpolant_marginal_allsky.pdf")

    import pickle
    filename='Marginalized_interpolation_allsky.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(interp_figaro_m, f)

conditional_density=True
if conditional_density:
    dpgmm_file = 'conditioned_density_draws_M1_and_DL.pkl' 

    draws = load_density(dpgmm_file)
    print([d.pdf([100,2500]) for d in draws])
    #bounds over which to interpolate for the parameters
    M_1=np.linspace(0,300,202)[1:-1]
    D_L=np.linspace(0,10000,202)[1:-1]
    #print(M_1.shape, D_L.shape)
    MM, DD = np.meshgrid(M_1, D_L)

    draws_pdf = np.log(np.mean([d.pdf(np.array([MM.flatten(), DD.flatten()]).T) for d in draws], axis = 0).reshape(len(D_L), len(M_1)) )

    interp_figaro_n= interp2d(M_1, D_L, draws_pdf, bounds_error=False)
    print("norm", interp_figaro_n(100,2500))
    plt.contourf( DD, MM, interp_figaro_n(M_1, D_L))
    # corner(M_1, D_L, draws[0].pdf(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1), len(D_L)) )
    plt.savefig("checking_interpolant_conditional.pdf")

    import pickle
    filename='conditional_interpolation.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(interp_figaro_n, f)

norm_compare=True
if norm_compare:
    dpgmm_file = 'conditioned_density_draws_M1_and_DL_nF.pkl' #detector frame M_1 and DL

    draws = load_density(dpgmm_file)
    #print([d.pdf([100,2500]) for d in draws])
    #bounds over which to interpolate for the parameters
    M_1=np.linspace(0,300,202)[1:-1]
    D_L=np.linspace(0,10000,202)[1:-1]
    #print(M_1.shape, D_L.shape)
    MM, DD = np.meshgrid(M_1, D_L)

    draws_pdf = np.log(np.mean([d.pdf(np.array([MM.flatten(), DD.flatten()]).T) for d in draws], axis = 0).reshape(len(D_L), len(M_1)) )

    interp_figaro_nF= interp2d(M_1, D_L, draws_pdf, bounds_error=False)
    print("nF", interp_figaro_nF(100,2500))
    plt.contourf( DD, MM, interp_figaro_nF(M_1, D_L))
    # corner(M_1, D_L, draws[0].pdf(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1), len(D_L)) )
    plt.savefig("checking_interpolant_conditional_nF.pdf")

    import pickle
    filename='conditional_interpolation_nF.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(interp_figaro_nF, f)

condition_4d=True
if condition_4d:
    
    filepath= 'primarymass/draws_GW190521.pkl'

    #the conditioned dist. 
    ra_EM, dec_EM = 192.42625 , 34.8247
    ra_EM_rad= ra_EM/360*np.pi*2
    dec_EM_rad=dec_EM/180*np.pi 

    draws = load_density(filepath)
    print([d.pdf([100,2500, ra_EM_rad,dec_EM_rad]) for d in draws])
    #bounds over which to interpolate for the parameters
    M_1=np.linspace(1,299,202)[1:-1]
    D_L=np.linspace(1,9999,202)[1:-1]
    #print(M_1.shape, D_L.shape)
    MM, DD = np.meshgrid(M_1, D_L)

    draws_pdf = np.log(np.mean([d.pdf(np.array([MM.flatten(), DD.flatten(), np.ones(len(MM.flatten()))*ra_EM_rad, np.ones(len(MM.flatten()))*dec_EM_rad]).T) for d in draws], axis = 0).reshape(len(D_L), len(M_1)) )
    #print(draws_pdf)
    interp_figaro_4= interp2d(M_1, D_L, draws_pdf, bounds_error=False)
    print("4d", interp_figaro_4(100,2500))
    plt.contourf( DD, MM, interp_figaro_4(M_1, D_L))
    # corner(M_1, D_L, draws[0].pdf(np.array([MM.flatten(), DD.flatten()]).T).reshape(len(M_1), len(D_L)) )
    plt.savefig("checking_interpolant_conditional_nF_4d.pdf")

    import pickle
    filename='conditional_interpolation_nF_4d.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(interp_figaro_4, f)
