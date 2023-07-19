
import numpy as np
import h5py
 
#no association, pl+pk prior
with h5py.File('inference_noEM_plpk/raynest.h5', 'r') as f:
    no_EM_plpk_logZ=  np.array(f['combined']['logZ'])


#redshift model
with h5py.File('inference/raynest.h5', 'r') as f:
    redshift_logZ=  np.array(f['combined']['logZ'])


print("estimated logZ for no EM counterpart and pl+pk model = {0} ".format(no_EM_plpk_logZ))
print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
print("Log Bayes' Factor redshift model vs nonredshifted= ", redshift_logZ - no_EM_plpk_logZ) #- not / for log"

comparerprior=True
if comparerprior==True:
    #redshift model with uniform prior on r
    with h5py.File('inference_norprior/raynest.h5', 'r') as f:
        nor_logZ = np.array(f['combined']['logZ'])
    print("estimated logZ for no prior on r (redshift model)= {0} ".format(nor_logZ))
    print("estimated logZ for prior on r (redshift model) = {0} ".format(redshift_logZ))
    print("Log Bayes' Factor r prior vs no r prior redshift model", redshift_logZ-nor_logZ) 