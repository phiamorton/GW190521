import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace



def prob(logslope, intercept, width1, width2, peakheight1, peakheight2, center1, center2, logr):
    linear= logslope * logr +intercept
    #normalize it in log space
    area= logslope*logr.max()**2+intercept*logr.max() - (logslope*logr.min()**2+intercept*logr.min())
    linear/=area
    #c=2 the scipy documentation is a lie
    peak1= laplace.logpdf(logr, center1, width1/10)
    peak2=laplace.logpdf(logr, center2, width2/10)
    return linear #+ peak1 +peak2)/3 # /3 because each is normalized 

logslope= 1
intercept=1

center1=0.85
center2=1.61
width1=np.log(2)
width2=np.log(2) #2 R_s on log scale
peakheight1= 0.5
peakheight2= 0.5

#radius (in terms of Swarzchild radii) on a log scale
logr= np.linspace(0.5, 4, 100000) #already log
print(logr)
#print(help(laplace.ppf))
#plt.plot(logr, prob(logslope, intercept, width1, width2, peakheight1, peakheight2, center1, center2, logr))

test1=laplace.pdf(logr,2,width1/100)
plt.plot(logr, )
plt.xlabel(r'log($\frac{r}{R_s}$)')
plt.ylabel('prob(BBH location)')
plt.show()

