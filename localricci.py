""" Coarse Ricci flow for a point cloud. """
import numpy as np


# treat some numpy warnings as errors?
np.seterr(all="print")  # divide='raise', invalid='raise')

#
#   simulation parameters
#
runs = 2000  # how many iterations
show = 20  # how frequently we show the result
eta = 0.0075  # factor of Ricci that is added to distance squared
threshold = 0.0001  # clustering threshold
# upperthreshold = 0.3  # won't try to cluster if distances in ambiguity interva (threshold, upperthreshold)
# 'min' rescales the distance squared function so minimum is 1.
# 'L1' rescales it so the sum of distance squared stays the same
#   (perhaps this is a misgnomer and it should be 'L2' but whatever)
# 'L_inf' rescales each to have diameter 1"
rescale = 'L_inf'
t = 0.3 # should not be integer to avaoid division problems.  This scale is used for computing the Laplace operator
T = 0.1 # scale used for localization of ricci flow
noise = 0.06  # noise coefficient
CLIP = 60  # value at which we clip distance function

np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(threshold=np.nan)

from tools import sanitize, is_clustered, color_clusters, is_stuck, components
from Laplacian import Laplacian
from Ricci import coarseRicci, applyRicci, getScalar
from data import noisycircles, noisymoons, two_clusters, perm_moons_200, perm_circles_200, four_clusters_3d
import matplotlib.pyplot as plt


# import data
# sqdist, pointset = data.two_clusters(35, 25, 2, dim=2)

n_samples = 20


def graph(threshold, mode="sort"):
    """
    Draw pointset as colored connected components.

    By default each component has a different color based on cluster number.

    They can also be colored using distance to one of the points of one of them.
    However, this does not work well for more than two clusters, since distances
    between clusters are about 1. (mode="dist")

    Components can also be sorted according to the distance to a point from one
    of them, to make less random looking coloring. (mode="sort")

    Distances are measured between representatives of the component, so close
    components may end up having large distance.
    """
    global ax
    c = np.zeros(len(pointset), dtype=int)
    num = components(sqdist, 0.001, c)
    print "Number of components: ", num
    # now c contains uniquely numbered components

    # replace colors with distance to a point
    # component number and a representative
    values, points = np.unique(c, return_index=True)
    dists = sqdist[points[0], points]
    if mode == "dist":
        c = dists[c]
    elif mode == "sort":
        a = sorted(zip(values, dists), key=lambda e:e[1])
        a = np.array(a)[:, 0]
        c = a[c]


    if len(points) < 10:
        print "Distances between components: \n", sqdist[np.ix_(points, points)]
    else:
        print "Distances to component 0: \n", dists
    plt.cla()
    if dim == 2:
        ax.scatter(pointset[:, 0], pointset[:, 1],  c=c, cmap='gnuplot2')
        plt.axis('equal')
    elif dim == 3:
        # from mpl_toolkits.mplot3d import Axes3D
        ax.scatter(pointset[:, 0], pointset[:, 1], pointset[:, 2],  c=c,
                   cmap='gnuplot2')
    plt.draw()
    plt.pause(0.01)
    return num


sqdist, pointset = noisymoons(500, noise)

# sqdist, pointset = two_clusters(500, 250, 7)
# sqdist, pointset = perm_circles_200()
# sqdist, pointset = four_clusters_3d(100,7)
dim = len(pointset[0])

fig = plt.figure()
if dim == 2:
    ax = fig.add_subplot(1, 1, 1)
else:
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')

#
#       !!! Memory management !!!
#
# We are dealing with many large matrices, but it seems that at most 5 must
# exist at any given time. And the limit is reached by coarseRicci.
#
# It might be best to preallocate them and keep reusing the memory.
# So no function should create any matrices (vectors are ok).
# Instead functions should request appropriate number of temporary and out
# arguments.
#
# E.g. Laplacian has an out argument, while Ricci has out and 2 temps.
#
# This rather extreme policy will also speed up the code, since no large memory
# will need to be allocated in the main loop.
#

L = oldsqdist = np.zeros_like(sqdist)
Ricci = np.zeros_like(sqdist)
mat1 = np.zeros_like(sqdist)
mat2 = np.zeros_like(sqdist)


sanitize(sqdist, rescale, np.inf, 1.0, temp=mat1)

graph(threshold)

Laplacian(sqdist, t, L)
coarseRicci(L, sqdist, Ricci, mat1, mat2)

# print 'initial distance'
# print sqdist
# print 'initial Ricci'
# print Ricci

applyRicci(sqdist, eta, T, Ricci, mode='sym')
sanitize(sqdist, rescale, CLIP, 1.0, temp=mat1)

graph(threshold)

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
        numclusters = graph(threshold, mode="sort")
        if is_clustered(sqdist, 10*threshold):
            break
        # if i>show and is_stuck(sqdist, oldsqdist, eta):  #it was getting falsely stuck very in some situations
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

if dim == 2:
    plt.scatter(pointset[:, 0], pointset[:, 1],  color=colors)
    plt.axis('equal')
    plt.show()
elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = pointset
    ax.scatter(data[:,0], data[:,1], data[:, 2],  color=colors)
    plt.show()
