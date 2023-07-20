
import numpy as np
import h5py
 
#no association, pl+pk prior
with h5py.File('inference_noEM_plpk/raynest.h5', 'r') as f:
    no_EM_plpk_logZ=  np.array(f['combined']['logZ'])

#pl+pk without tapering
with h5py.File('inference_no_tapering/raynest.h5', 'r') as f:
    no_EM_plpk_no_tapering_logZ=  np.array(f['combined']['logZ'])


#redshift model with r prior
with h5py.File('inference_M1_rprior_interp/raynest.h5', 'r') as f:
    redshift_logZ=  np.array(f['combined']['logZ'])


print("estimated logZ for no EM counterpart and pl+pk model = {0} ".format(no_EM_plpk_logZ))
print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
print("Log Bayes' Factor redshift model vs no EM association pl+pk= ", redshift_logZ - no_EM_plpk_logZ) #- not / for log"


print("estimated logZ for no EM counterpart and pl+pk model without tapering= {0} ".format(no_EM_plpk_no_tapering_logZ))
#print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
print("Log Bayes' Factor redshift model vs no EM association pl+pk without tapering= ", redshift_logZ - no_EM_plpk_no_tapering_logZ) #- not / for log"

#print("estimated logZ for no EM counterpart and pl+pk model with tapering = {0} ".format(no_EM_plpk_logZ))
#print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
print("Log Bayes' Factor no EM association pl+pk tapered vs untapered= ", no_EM_plpk_logZ - no_EM_plpk_no_tapering_logZ) #- not / for log"

comparerprior=True
if comparerprior:
    #redshift model with uniform prior on r
    with h5py.File('inference_norprior_M1_interp/raynest.h5', 'r') as f:
        nor_logZ = np.array(f['combined']['logZ'])
    print("estimated logZ for no prior on r (redshift model)= {0} ".format(nor_logZ))
    print("estimated logZ for prior on r (redshift model) = {0} ".format(redshift_logZ))
    print("Log Bayes' Factor r prior vs no r prior redshift model", redshift_logZ-nor_logZ) 