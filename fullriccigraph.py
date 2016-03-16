""" Coarse Ricci flow for a point cloud. """
import numpy as np
import numexpr as ne
# treat some numpy warnings as errors?
np.seterr(all="print")  # divide='raise', invalid='raise')

#
#   simulation parameters
#
runs = 10000  # how many iterations
show = 1000  # how frequently we show the result
eta = 0.0002  # factor of Ricci that is added to distance squared
# 'min' rescales the distance squared function so minimum is 1.
# 'L1' rescales it so the sum of distance squared stays the same
#   (perhaps this is a misgnomer and it should be 'L2' but whatever)
rescale = 'L1'
t = 0.1  # should not be integer to avaoid division problems
noise = 0.2  # noise coefficient
CLIP = 60  # value at which we clip distance function


import data
from tools import metricize
from Laplacian import Laplacian
from Ricci import coarseRicci


# sqdist = data.onedimensionpair(2, 3, noise)
# sqdist = data.cyclegraph(6, noise)
sqdist = data.closefarsimplices(3, 0.1, 3)

sqdist = metricize(sqdist)
L = Laplacian(sqdist, t)
Ricci = coarseRicci(L, sqdist)

print 'initial distance'
print sqdist
print 'initial Ricci'
print Ricci


ne.evaluate("sqdist-eta*Ricci", out=sqdist)

initial_L1 = sqdist.sum()

for i in range(runs + show + 3):
    L = Laplacian(sqdist, t)
    Ricci = coarseRicci(L, sqdist)
    ne.evaluate("sqdist-eta*Ricci", out=sqdist)
    sqdist = ne.evaluate("(sqdist + sqdistT)/2",
                         global_dict={'sqdistT': sqdist.transpose()})

    # total_distance = sqdist.sum()
    # sqdist = (total_distance0/total_distance)*sqdist
    nonzero = sqdist[np.nonzero(sqdist)]
    mindist = np.amin(nonzero)
    s1 = mindist
    s2 = sqdist.sum()
    # print t
    # ne.evaluate("dist/s", out=dist)

    sqdist = np.clip(sqdist, 0, CLIP)
    if rescale == 'L1':
        ne.evaluate("initial_L1*sqdist/s2", out=sqdist)
    if rescale == 'min':
        ne.evaluate("sqdist/s1", out=sqdist)
    dist = metricize(sqdist)
    if i % show == 2:
        # print Ricci
        print "sqdist for ", i, "  time"
        print sqdist
        print 't = ', t
        # print Ricci
        # print Ricci/dist, '<- Ricc/dist'
        print '---------'
