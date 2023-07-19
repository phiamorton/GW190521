import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace



def rad_prior(r):
    #linear in log scale:
    #logslope= 1
    slope= 0.01 
    intercept=1

    #Laplacian/resonance peaks:
    center1=330
    center2=24.5   #https://arxiv.org/pdf/1511.00005.pdf 
    width1=5*np.log(2)
    width2=5*np.log(2) #2 R_s on log scale
    linear= slope * r +intercept
    #normalize it in log space??????
    #area= logslope*logr.max()**2+intercept*logr.max() - (logslope*logr.min()**2+intercept*logr.min())
    #linear/=area

    #c=2 the scipy documentation is a lie
    #defined with a center and scale in log space
    #just eyeballed scale factor to shrink the width and then rescaled the entire distribution 
    peak1= (laplace.pdf(r, center1, width1)) *50
    peak2=(laplace.pdf(r, center2, width2)) *50
    return linear + peak1 +peak2      # /3 if each is normalized IF WANT TO NORMALIZE? MAYBE?

if __name__ == '__main__':

    #radius (in terms of Swarzchild radii) on a log scale
    r= np.linspace(1, 400, 100000) #already log
    #print(logr)

    plt.plot(r, rad_prior(r))

    #print(help(laplace.ppf))
    #test1=laplace.pdf(logr,2,width1/100)
    #plt.plot(logr, test1)
    plt.xlabel(r'$\frac{r}{R_s}$')
    plt.ylabel('prob(BBH location)')
    plt.show()

