""" Coarse Ricci flow for a point cloud. """
import numpy as np
import numexpr as ne


# treat some numpy warnings as errors?
np.seterr(all="print")  # divide='raise', invalid='raise')

#
#   simulation parameters
#
runs = 2000  # how many iterations
show = 10  # how frequently we show the result
eta = 0.0075  # factor of Ricci that is added to distance squared
threshold = 0.05  # clustering threshold
upperthreshold = 0.6  # won't try to cluster if distances in ambiguity interva (threshold, upperthreshold)
# 'min' rescales the distance squared function so minimum is 1.
# 'L1' rescales it so the sum of distance squared stays the same
#   (perhaps this is a misgnomer and it should be 'L2' but whatever)
# 'L_inf' rescales each to have diameter 1"
rescale = 'L1'
t = 0.4 # should not be integer to avaoid division problems.  This scale is used for computing the Laplace operator
T = 0.1 # scale used for localization of ricci flow
noise = 0.05  # noise coefficient
CLIP = 60  # value at which we clip distance function

np.set_printoptions(precision=2, suppress=True)

from tools import sanitize, is_clustered, color_clusters, is_stuck
from Laplacian import Laplacian
from Ricci import coarseRicci, applyRicci
from data import noisycircles, noisymoons


# import data
# sqdist, pointset = data.two_clusters(35, 25, 2, dim=2)
twodim = True

n_samples = 120



#pointset, sqdist = noisycircles(n_samples, .5, noise)

sqdist, pointset = noisymoons(n_samples, noise)

sanitize(sqdist)
L = Laplacian(sqdist, t)
Ricci = coarseRicci(L, sqdist)

print 'initial distance'
print sqdist
print 'initial Ricci'
print Ricci

loosekernel = ne.evaluate('eta*exp(-sqdist/T)')
applyRicci(sqdist, loosekernel, Ricci, mode='sym')

initial_L1 = sqdist.sum()
# This will modify Ricci locally more than far away.
clustered = False
oldsqdist = np.copy(sqdist)
for i in range(runs + show + 3):
    loosekernel[:] = ne.evaluate('eta*exp(-sqdist/T)')
    L[:] = Laplacian(sqdist, t)
    Ricci[:] = coarseRicci(L, sqdist)
    applyRicci(sqdist, loosekernel, Ricci, mode='sym')

    # total_distance = sqdist.sum()
    # sqdist = (total_distance0/total_distance)*sqdist
    # print t
    # ne.evaluate("dist/s", out=dist)

    sanitize(sqdist, CLIP, initial_L1)

    if i % show == 2:
        print Ricci
        print "sqdist for ", i, "  time"
        print sqdist
        print 't = ', t
        # print Ricci
        # print Ricci/dist, '<- Ricc/dist'
        print '---------'
        if is_stuck(sqdist,oldsqdist,eta):
            print 'stuck!'
            clustered
            break
        oldsqdist = np.copy(sqdist)
        if ne.evaluate('(sqdist>threshold)&(sqdist<upperthreshold)').any():
            print 'values still in ambiguous interval'
            continue
        if is_clustered(sqdist, threshold):
            clustered = True
            clust = color_clusters(sqdist, threshold)
            break


if not clustered :
    if is_clustered(sqdist, threshold):
         clust = color_clusters(sqdist, threshold)
    else :
        print 'not clustered at all'
        quit()


choices = 'rgbmyc'
colors = [(choices[j] if j < len(choices) else 'k') for j in clust]
print clust
print colors

if twodim:
    import matplotlib.pyplot as plt
    plt.scatter(pointset[:, 0], pointset[:, 1],  color=colors)
    plt.axis('equal')
    plt.show()
