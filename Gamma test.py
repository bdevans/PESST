from __future__ import division
import numpy as np
from numpy import median
import scipy.special as sps
from numpy.random import gamma
from matplotlib import pyplot as plt
from tqdm import tqdm
from textwrap import wrap

# user defined settings
iterations = 100
gammasamples = 10000

# user defined gamma. Seems a lot of systems only let you set kappa (often called shape alpha) and calculate theta as 1/kappa giving mean of 1
shape = 1.9
scale = 1/shape

#space for calculations of iterations
medians=[]
medianlower=[]
medianlowermid=[]
medianuppermid=[]
medianupper=[]

bottomquarts=[]
bottommidquarts=[]
topmidquarts=[]
topquarts=[]


# This loop makes a set number of gamma rate categories
# Does so by sampling many times from a gamma distribution
# Tests of the trade-off between computing time and variance led me to set this to 10000 samples from the distribution
# Computes quartiles from the data with equal likelihood by finding bounds for quartiles and collecting values between bounds.
# Finds discrete rate values by sorting collected values and taking the median
# The code then iterates for a predefined set of runs, recording the median values
# Tests of the tradeoff between computing time and variance led me to set this to 100 independent runs (1 million total samples from distribution). Takes between 1 and 2 seconds with an intel i5 processor and 8GB ram for variance between runs of <0.1%.

for i in tqdm(range(iterations)):

    # sample gamma
    s = gamma(shape, scale, gammasamples)

    # define quartiles in that data with equal probability
    bottomquart = np.percentile(s, [0,25], interpolation='midpoint')
    bottomquarts.append(bottomquart)
    bottommidquart = np.percentile(s, [25,50], interpolation='midpoint')
    bottommidquarts.append(bottommidquart)
    topmidquart = np.percentile(s, [50,75], interpolation='midpoint')
    topmidquarts.append(topmidquart)
    topquart = np.percentile(s, [75,100], interpolation='midpoint')
    topquarts.append(topquart)
    # print bottomquart, bottommidquart, topmidquart, topquart

    # generate space for the values within each quartile, sort them, find the median, record the median.
    bottomlist = []
    bottommidlist = []
    topmidlist = []
    toplist = []
    for i in s:
        if bottomquart[0] <= i < bottomquart[-1]:
            bottomlist.append(i)
        elif bottommidquart[0] <= i < bottommidquart[-1]:
            bottommidlist.append(i)
        elif topmidquart[0] <= i < topmidquart[-1]:
            topmidlist.append(i)
        else:
            toplist.append(i)
    bottomlist.sort()
    bottommidlist.sort()
    topmidlist.sort()
    toplist.sort()
    ratecategoriesquartile = [median(bottomlist), median(bottommidlist), median(topmidlist), median(toplist)]
    medians.append(ratecategoriesquartile)

    # print ratecategoriesquartile

#calculate average of medians from each iteration
for i in medians:
    medianlower.append(i[0])
    medianlowermid.append(i[1])
    medianuppermid.append(i[2])
    medianupper.append(i[3])

finalmedians = [np.mean(medianlower), np.mean(medianlowermid), np.mean(medianuppermid), np.mean(medianupper)]

#print finalmedians

#calculate mean quartile bounds from each iteration
bottomquartlowerbounds = []
bottomquartupperbounds = []
bottommidquartupperbounds = []
topmidquartupperbounds = []
topquartupperbounds = []

for i in bottomquarts:
    bottomquartlowerbounds.append(i[0])
    bottomquartupperbounds.append(i[-1])

for i in bottommidquarts:
    bottommidquartupperbounds.append(i[-1])

for i in topmidquarts:
    topmidquartupperbounds.append(i[-1])

for i in topquarts:
    topquartupperbounds.append(i[-1])


# plot the distribution as well as the quartiles and medians
xtoplot=[np.mean(bottomquartlowerbounds), np.mean(bottomquartupperbounds), np.mean(bottommidquartupperbounds), np.mean(topmidquartupperbounds), np.mean(topquartupperbounds)]
x=np.linspace(0,6,1000)
y = x**(shape-1)*(np.exp(-x/scale) / (sps.gamma(shape)*scale**shape))
plt.plot(x, y, linewidth=2, color='k', alpha=0)
plt.fill_between(x,y, where = x>xtoplot[0], color = '#4c4cff')
plt.fill_between(x,y, where = x>xtoplot[1], color = '#7f7fff')
plt.fill_between(x,y, where = x>xtoplot[2], color = '#b2b2ff')
plt.fill_between(x,y, where = x>xtoplot[3], color = '#e5e5ff')
plt.axvline(x=finalmedians[0],color="#404040", linestyle=":")
plt.axvline(x=finalmedians[1],color="#404040", linestyle=":")
plt.axvline(x=finalmedians[2],color="#404040", linestyle=":")
plt.axvline(x=finalmedians[3],color="#404040", linestyle=":")
plt.title("\n".join(wrap('gamma rate categories calculated as the the average of %s median values of 4 equally likely quartiles of %s randomly sampled vaules' % (iterations, gammasamples), 60)), fontweight='bold', fontsize = 10)
plt.text(5, 0.6, "$\kappa$ = %s\n$\\theta$ = $\\frac{1}{\kappa}$" % (shape))
plt.show()