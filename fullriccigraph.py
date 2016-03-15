#!/Users/siudeja/anaconda/bin/python
""" Coarse Ricci flow for a point cloud. """
import numpy as np
import numexpr as ne
from scipy.misc import logsumexp
from numba import jit, vectorize
import timeit

n = 3
t = 0.1  # should not be integer to avaoid division problems
noise = 0.1  # expansion coefficient
# treat some numpy warnings as errors
np.seterr(all="print")  # divide='raise', invalid='raise')


@jit("void(f8[:,:], f8)", nopython=True, nogil=True)
def symmetric(A, sigma):
    """
    Symmetric random normal matrix with -1 on the diagonal.

    Compiled using numba jit!
    """
    n = len(A) / 2
    # blocks around diagonal (symmetric, 0 diagonal at first)
    for i in range(n):
        for j in range(i+1, n):
            A[i, j] = A[j, i] = A[i+n, j+n] = A[j+n, i+n] = \
                np.random.normal(1.0, sigma)
    # off diagonal blocks 4*(diag block)+noise
    for i in range(n):
        for j in range(n):
            A[i, j+n] = A[j+n, i] = 4*A[i, j] + np.random.normal(1.0, sigma)
        # matrix diagonal adjusted last
        # A[i, i] = A[i+n, i+n] = -1.0


@vectorize("f8(f8, f8)")
def logaddexp(a, b):
    """ Vectorized logaddexp. """
    if a < b:
        return np.log1p(np.exp(a-b)) + b
    elif a > b:
        return np.log1p(np.exp(b-a)) + a
    else:
        return np.log(2.0) + a


def computeLaplaceMatrix2(dMatrix, t):
    """
    Compute heat approximation to Laplacian matrix using logarithms.

    This is slightly slower, but hopefully more accurate.

    No numexpr to catch numpy floating point errors.

    To compute L_t f just multiply L_t by vector f.
    """
    lt = np.log(2/t)
    L = dMatrix*dMatrix
    L /= -2.0*t
    # numpy floating point errors likely below
    logdensity = logsumexp(L, axis=1)
    # logdensity = logaddexp.reduce(np.sort(L, axis=1), axis=1)
    # compute log(density-1):
    # np.log(np.expm1(logdensity))
    # logdensity + np.log1p(-exp(logdensity))

    # sum in rows must be 1
    L = np.exp(L - logdensity[:, None] + lt)
    # fix diagonal to account for -f(x)?
    # L_t matrix is the unajusted one - scaled identity
    L[np.diag_indices(len(L))] -= 2.0/t
    # alternatively L_t could be computed using unadjusted matrix
    # applied to f - f at point
    return L


def coarseRicci(L, dMatrix):
    """ numexpr parallelized Ricci. """
    Ric = np.zeros_like(dMatrix)
    for i in xrange(len(L)):
        # now f_i and cdc are matrices
        f_i = ne.evaluate("ci-dMatrix",
                          global_dict={'ci': dMatrix[:, i, None]})
        # first CDC
        cdc = L.dot(ne.evaluate("f_i*f_i"))
        Lf = L.dot(f_i)
        # end of CDC1 combined with CDC2
        ne.evaluate("cdc/2.0-2.0*f_i*Lf", out=cdc)
        cdc = L.dot(cdc)
        ne.evaluate("cdc+f_i*LLf+Lf*Lf",
                    global_dict={'LLf': L.dot(Lf)}, out=cdc)
        Ric[i, :] = cdc[i]/2.0
    return Ric


def coarseRicci3(L, dMatrix):
    """
    Precompute Ld first and try to avoid mat-mat multiplications.

    This one is about 3x faster.
    """
    Ric = np.zeros_like(dMatrix)
    Ld = L.dot(dMatrix)
    for i in xrange(len(L)):
        # now f_i and cdc are matrices
        f_i = ne.evaluate("di-dMatrix",
                          global_dict={'di': dMatrix[:, i, None]})
        # first CDC
        # FIXME how to compute L(di*d) quickly ??
        # this one is the only matrix-matrix multiplication
        cdc = L.dot(ne.evaluate("f_i*f_i"))
        # L is linear so Lf = Ld[:, i, None] - Ld
        Lf = ne.evaluate("Ldi-Ld", global_dict={'Ldi': Ld[:, i, None]})
        # end of CDC1 combined with CDC2
        ne.evaluate("cdc-4.0*f_i*Lf", out=cdc)
        # we are using one row from cdc in Ric, so we can use one row from L
        cdc = L[i].dot(cdc)
        # we can also use one row from the rest too
        ne.evaluate("(cdc/2.0+f_ii*LLfi+Lfi*Lfi)/2.0",
                    global_dict={
                        'LLfi': L[i].dot(Lf),
                        'Lfi': Lf[i],
                        'f_ii': f_i[i]
                    }, out=cdc)
        Ric[i, :] = cdc
    return Ric

# def CDC1(L,f,g):
#     u=f*g
#     cdc = L.dot(u)-f*L.dot(g)-g*L.dot(f)
#     return cdc/2
#
# def coarseRicci2(L,dMatrix):
#     Ric = np.zeros((len(dMatrix),len(dMatrix)))
#     for i in range(len(L)):
#         for j in range(len(L)):
#             f_ij = dMatrix[:,i]-dMatrix[:,j]
#             cdc1 = CDC1(L, f_ij,f_ij)
#             cdc2 = L.dot(cdc1)-2*CDC1(L,f_ij,L.dot(f_ij))
#             Ric[i,j]=cdc2[i]/2
#     return Ric


def test(f, args_string):
    """ Test speed of a function. """
    print f.__name__
    t = timeit.repeat("%s(%s)" % (f.__name__, args_string),
                      repeat=5, number=1,
                      setup="from __main__ import %s, %s" % (f.__name__,
                                                             args_string))
    print min(t)


eta = .0001
dist = np.zeros((2*n, 2*n))
symmetric(dist, eta)
# FIXME why is dist matrix -1 on diagonal?


print computeLaplaceMatrix2(dist, t)
test(computeLaplaceMatrix2, "dist, t")

exit(0)
L = computeLaplaceMatrix2(dist, t)

print "test Ricci, next line should be 0"
print np.max(np.abs(coarseRicci(L, dist)-coarseRicci3(L, dist)))
test(coarseRicci, "L, dist")
test(coarseRicci3, "L, dist")

exit(0)

Ricci = coarseRicci3(L, dist)
print Ricci


total_distance0 = dist.sum()


print 'initial distance'
print dist
ne.evaluate("dist-eta*Ricci", out=dist)
print 'new dist'
print dist
c = 1

for i in range(5):
    L = computeLaplaceMatrix2(dist, t)
    Ricci = coarseRicci3(L, dist)
    ne.evaluate("dist-eta*Ricci", out=dist)
    print dist
    dist = ne.evaluate("(dist + distT)/2",
                       global_dict={'distT': dist.transpose()})

    # total_distance = dist.sum()
    # dist = (total_distance0/total_distance)*dist
    nonzero = dist[np.nonzero(dist)]
    mindist = np.amin(nonzero)
    t = mindist
    # print t
    ne.evaluate("dist/t", out=dist)
    if i % 900 == 2:
        # print Ricci
        print dist
        # print Ricci/dist
        print '---------'
