""" Coarse Ricci flow for a point cloud. """
import numpy as np


# treat some numpy warnings as errors?
np.seterr(all="print")  # divide='raise', invalid='raise')

#
#   simulation parameters
#
runs = 2000  # how many iterations
show = 20  # how frequently we show the result
eta = 0.0015  # factor of Ricci that is added to distance squared
# do not cluster if distances in ambiguity interval (threshold, upperthreshold)
threshold = 0.0001  # clustering threshold
# upperthreshold = 0.3
# 'min' rescales the distance squared function so minimum is 1.
# 'L1' rescales it so the sum of distance squared stays the same
#   (perhaps this is a misgnomer and it should be 'L2' but whatever)
# 'L_inf' rescales each to have diameter 1"
rescale = 'L_inf'
t = 0.3  # This scale is used for computing the Laplace operator
T = 0.1  # scale used for localization of ricci flow
noise = 0.06  # noise coefficient
CLIP = 60  # value at which we clip distance function

# make sure integer division does not happen
t = float(t)
T = float(T)
noise = float(noise)

np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(threshold=np.nan)

from tools import sanitize, is_clustered, color_clusters, get_matrices, \
    init_plot, graph
from Laplacian import Laplacian
from Ricci import coarseRicci, applyRicci  # , getScalar
import data


# sqdist, pointset = data.two_clusters(35, 25, 2, dim=2)
sqdist, pointset = data.noisymoons(1000, noise)
# sqdist, pointset = data.two_clusters(500, 250, 7)
# sqdist, pointset = data.perm_circles_200()
# sqdist, pointset = data.four_clusters_3d(100,7)
dim = len(pointset[0])
plt, ax = init_plot(dim)

#
#       !!! Memory management !!!
#
# We are dealing with many large matrices, but it seems that at most 5 must
# exist at any given time. And the limit is reached by coarseRicci.
#
# It might be best to preallocate them and keep reusing the memory.
# So no function should create any matrices (vectors are ok).
# Instead functions should request appropriate number of temporary matrices.
#
# E.g. Laplacian has an outpu L argument, while Ricci has output R and 2 temps.
#
# This rather extreme policy will also speed up the code, since no large memory
# will need to be allocated in the main loop.
#
# Furthermore, we can ensure that all matrices are properly aligned for
# vectorized SIMD operations.
#

# 5 matrices
L, Ricci, mat1, mat2, sqdist = get_matrices(sqdist, 4)
oldsqdist = L

sanitize(sqdist, rescale, np.inf, 1.0, temp=mat1)

graph(threshold, pointset, sqdist, ax, dim)

Laplacian(sqdist, t, L)
coarseRicci(L, sqdist, Ricci, mat1, mat2)

# print 'initial distance'
# print sqdist
# print 'initial Ricci'
# print Ricci

applyRicci(sqdist, eta, T, Ricci, mode='sym')
sanitize(sqdist, rescale, CLIP, 1.0, temp=mat1)

graph(threshold, pointset, sqdist, ax, dim)

clustered = False

# L is the same as sqdist
for i in range(runs + show + 3):
    Laplacian(sqdist, t, L)
    coarseRicci(L, sqdist, Ricci, temp1=mat1, temp2=mat2)
    # now Laplacian is useless, so oldsqdist can replace it
    np.copyto(oldsqdist, sqdist)
    applyRicci(sqdist, eta, T, Ricci, mode='sym')

    # total_distance = sqdist.sum()
    # sqdist = (total_distance0/total_distance)*sqdist
    # print t
    # ne.evaluate("dist/s", out=dist)

    sanitize(sqdist, rescale, CLIP,  1.0, temp=mat1)

    if i % show == 2:
        # print Ricci
        # print "sqdist for ", i, "  time"
        # print sqdist
        # print 't = ', t
        # print Ricci
        # print Ricci/dist, '<- Ricc/dist'
        # print '---------'
        # scalar = getScalar(Ricci, sqdist, t)
        # print np.std(scalar), scalar.mean(), np.std(scalar)/scalar.mean()
        # print scalar
        # print '##########'
        numclusters = graph(threshold, pointset, sqdist, ax, dim)
        if is_clustered(sqdist, 10*threshold):
            break
        # if i>show and is_stuck(sqdist, oldsqdist, eta):
        #     it was getting falsely stuck very in some situations
        #     print 'stuck!'
        #     clustered
        #     break
        # oldsqdist = np.copy(sqdist)
        # if ne.evaluate('(sqdist>threshold)&(sqdist<upperthreshold)').any():
        #     print 'values still in ambiguous interval'
        #     continue
        # if is_clustered(sqdist, threshold):
        #     clustered = True
        #     clust = color_clusters(sqdist, threshold)
        #     break

# exit(0)

if not clustered:
    if is_clustered(sqdist, 10*threshold):
        clust = color_clusters(sqdist, 10*threshold)
    else:
        print 'not clustered at all'
        quit()


choices = 'rgbmyc'
colors = [(choices[j] if j < len(choices) else 'k') for j in clust]
print clust
print colors

plt.cla()
if dim == 2:
    ax.scatter(pointset[:, 0], pointset[:, 1],  color=colors)
    plt.show()
elif dim == 3:
    ax.scatter(pointset[:, 0], pointset[:, 1], pointset[:, 2],  color=colors)
    plt.show()
