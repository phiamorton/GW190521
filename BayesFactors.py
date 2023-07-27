
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

#with free H_0
with h5py.File('inference_H/raynest.h5', 'r') as f:
    H_0_logZ=  np.array(f['combined']['logZ'])

with h5py.File('inference_H_GW17/raynest.h5', 'r') as f:
    H_0_withGW17 = np.array(f['combined']['logZ'])

print("Log Bayes' factor with and without GW170817 prior{:.2f}".format( H_0_withGW17-H_0_logZ))

# print("\nestimated logZ for no EM counterpart and pl+pk model = {0} ".format(no_EM_plpk_logZ))
# print("\nestimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
print("\nLog Bayes' Factor redshift model vs no EM association pl+pk= {:.2f}".format(redshift_logZ - no_EM_plpk_logZ ) )#- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"

#H_0
# print("\nestimated logZ for redshift model with free H_0 = {0} ".format(H_0_logZ))
# print("\nLog Bayes' Factor redshift model vs redshift model with free H_0= ", redshift_logZ - H_0_logZ ) #- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"
print("\nLog Bayes' Factor redshift model with free H_0 vs no EM association pl+pk= {:.2f}".format(H_0_logZ - no_EM_plpk_logZ )) #- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"

#H_0 and GW170817
# print("\nestimated logZ for redshift model with free H_0 and GW170817 = {0} ".format(H_0_withGW17))
# print("\nLog Bayes' Factor redshift model vs redshift model with free H_0 and GW170817= ", redshift_logZ - H_0_withGW17 ) #- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"
print("\nLog Bayes' Factor redshift model with free H_0 and GW170817 vs no EM association pl+pk= {:.2f}".format( H_0_withGW17 - no_EM_plpk_logZ )) #- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"


# print("\nestimated logZ for no EM counterpart and pl+pk model without tapering= {0} ".format(no_EM_plpk_no_tapering_logZ))
# #print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
# print("\nLog Bayes' Factor redshift model vs no EM association pl+pk without tapering= ", redshift_logZ - no_EM_plpk_no_tapering_logZ ) #- np.log(13)) #- not / for log"

# #print("estimated logZ for no EM counterpart and pl+pk model with tapering = {0} ".format(no_EM_plpk_logZ))
# #print("estimated logZ for redshift model with r prior= {0} ".format(redshift_logZ))
# print("\nLog Bayes' Factor no EM association pl+pk tapered vs untapered= ", no_EM_plpk_logZ - no_EM_plpk_no_tapering_logZ) #- not / for log"

comparerprior=False
if comparerprior:
    #redshift model with uniform prior on r
    with h5py.File('inference_norprior_M1_interp/raynest.h5', 'r') as f:
        nor_logZ = np.array(f['combined']['logZ'])
    print("\nestimated logZ for no prior on r (redshift model)= {0} ".format(nor_logZ))
    print("\nestimated logZ for prior on r (redshift model) = {0} ".format(redshift_logZ))
    print("\nLog Bayes' Factor r prior vs no r prior redshift model", redshift_logZ-nor_logZ) 


remove_prior=True
Ashton_prior=np.log(13)
if remove_prior:
    print("\nLog Bayes' Factor redshift model vs no EM association pl+pk -Ashton prior={:.2f} ".format( redshift_logZ - no_EM_plpk_logZ -Ashton_prior) )#- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"

    print("\nLog Bayes' Factor redshift model with free H_0 vs no EM association pl+pk -Asthon prior= {:.2f}".format(H_0_logZ - no_EM_plpk_logZ -Ashton_prior)) #- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"
    print("\nLog Bayes' Factor redshift model with free H_0 and GW170817 vs no EM association pl+pk- Asthon prior= {:.2f}".format( H_0_withGW17 - no_EM_plpk_logZ -Ashton_prior) )#- np.log(13)) #if subtracting prior odds as in Ashton#- not / for log"
    