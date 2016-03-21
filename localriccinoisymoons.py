""" Coarse Ricci flow for a point cloud. """
import numpy as np
import numexpr as ne


from sklearn import cluster, datasets


# treat some numpy warnings as errors?
np.seterr(all="print")  # divide='raise', invalid='raise')

#
#   simulation parameters
#
runs = 100000  # how many iterations
show = 10  # how frequently we show the result
eta = 0.0075 # factor of Ricci that is added to distance squared
threshold = 0.05 #clustering threshold
upperthreshold = .45 # won't try to cluster if distances in ambiguity interva (threshold, upperthreshold)
T = .3 #this is the "outer scale"

# 'min' rescales the distance squared function so minimum is 1.
# 'L1' rescales it so the sum of distance squared stays the same
#   (perhaps this is a misgnomer and it should be 'L2' but whatever)
rescale = 'L1'
t = 0.05  # should not be integer to avaoid division problems
noise = 0.015 # noise coefficient
CLIP = 60  # value at which we clip distance function

np.set_printoptions(precision=2,suppress = True)


import data
from tools import metricize
from tools import is_clustered, color_clusters
from Laplacian import Laplacian
from Ricci import coarseRicci


twodim=True

n_samples = 200
k = 14
pointset,Zcolors = datasets.make_moons(n_samples=n_samples,noise=noise)
#farnoise = np.random.normal(size = (k,2))
#Z = np.concatenate((pointset,farnoise))
Z = pointset
print Z
n=len(Z)
sqdist = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sqdist[i,j]=(Z[i,0]-Z[j,0])*(Z[i,0]-Z[j,0])+(Z[i,1]-Z[j,1])*(Z[i,1]-Z[j,1])

    

metricize(sqdist)
L = Laplacian(sqdist, t)
Ricci = coarseRicci(L, sqdist)

print 'initial distance'
print sqdist
print 'initial Ricci'
print Ricci


ne.evaluate("sqdist-eta*Ricci", out=sqdist)

initial_L1 = sqdist.sum()
loosekernel = eta*np.exp(-sqdist/T) # This will modify Ricci locally more than far away. 
 

for i in range(runs + show + 3):

    L = Laplacian(sqdist, t)
    Ricci = coarseRicci(L, sqdist)
    ne.evaluate("sqdist-loosekernel*Ricci", out=sqdist)  # changes closer points now
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
    metricize(sqdist)
    if i % show == 2:
        print Ricci
        print "sqdist for ", i, "  time"
        print sqdist
        print 't = ', t
        # print Ricci
        # print Ricci/dist, '<- Ricc/dist'
        print '---------'
        if ((sqdist>threshold)&(sqdist<upperthreshold)).any():
            print 'values still in ambiguous interval'
            continue
        if is_clustered(sqdist,threshold):
            clust = color_clusters(sqdist,threshold)
            print clust
            break

n = len(clust)
colors = [None]*n
for j in range(n):
	if clust[j]==0: colors[j]='r'
	if clust[j]==1: colors[j]='g'
	if clust[j]==2: colors[j]='b'
	if clust[j]==3: colors[j]='m'
	if clust[j]==4: colors[j]='y'
	if clust[j]==5: colors[j]='c'
	if clust[j]>5: colors[j]='k'

if twodim:
	np.savetxt('Zcolors.csv',Zcolors)
	import matplotlib.pyplot as plt
	plt.scatter(Z[:,0], Z[:,1], color = colors)
	plt.axis('equal')
	plt.show()
	