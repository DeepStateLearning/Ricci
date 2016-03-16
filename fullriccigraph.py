#!/Users/siudeja/anaconda/bin/python
""" Coarse Ricci flow for a point cloud. """
import numpy as np
import numexpr as ne
import scipy.misc as sm
from numba import jit  # , vectorize
import timeit

n = 9
k = 4
l = 4
runs = 10000  # how many iterations
show = 1000  # how frequently we show the result
eta = 0.0002  # factor of Ricci that is added to distance squared
# 'min' rescales the distance squared function so minimum is 1.   'L1' rescales it so the sum of distance squared stays the same (perhaps this is a misgnomer and it should be 'L2' but whatever)
rescale = 'L1'
t = 0.1  # should not be integer to avaoid division problems
noise = 0.2  # noise coefficient
CLIP = 60  # value at which we clip distance function

import gmpy2 as mp
mp.get_context().precision = 200
exp = np.frompyfunc(mp.exp, 1, 1)
expm1 = np.frompyfunc(mp.expm1, 1, 1)
log = np.frompyfunc(mp.log, 1, 1)
is_finite = np.frompyfunc(mp.is_finite, 1, 1)
to_mpfr = np.frompyfunc(mp.mpfr, 1, 1)
to_double = np.frompyfunc(float, 1, 1)


# treat some numpy warnings as errors
np.seterr(all="print")  # divide='raise', invalid='raise')


# Note dist is always the distance squared matrix.

@jit("void(f8[:,:], f8)", nopython=True, nogil=True)
def symmetric(A, sigma):
    """
    Symmetric random normal matrix with -1 on the diagonal.

    Compiled using numba jit!
    """
    n = len(A) / 2
    # blocks around diagonal (symmetric, 0 diagonal at first)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = A[i + n, j + n] = A[j + n, i + n] = \
                np.random.normal(1.0, sigma)
    # off diagonal blocks 4*(diag block)+noise
    for i in range(n):
        for j in range(n):
            A[i, j + n] = A[j + n, i] = 4 * \
                A[i, j] + np.random.normal(1.0, sigma)


def logsumexp(a):
    """ mpfr compatible minimal logsumexp version. """
    m = np.max(a, axis=1)
    return log(np.sum(exp(a - m[:, None]), axis=1)) + m


def computeLaplaceMatrix(dMatrix, t, logeps=mp.mpfr("-10")):
    """
    Compute heat approximation to Laplacian matrix using logarithms.

    Use gmpy2 mpfr to gain more precision.

    This is slow, but more accurate.

    Cutoff for really small values, and row elimination if degenerate.
    """
    # cutoff ufunc
    cutoff = np.frompyfunc((lambda x: mp.inf(-1) if x < logeps else x), 1, 1)

    t2 = mp.mpfr(t)
    lt = mp.log(2 / t2)
    d = to_mpfr(dMatrix)
    L = d * d
    L /= -2 * t2
    cutoff(L, out=L)
    logdensity = logsumexp(L)
    L = exp(L - logdensity[:, None] + lt)
    L[np.diag_indices(len(L))] -= 2 / t2
    L = np.array(to_double(L), dtype=float)
    # if just one nonzero element, then erase row and column
    degenerate = np.sum(L != 0.0, axis=1) <= 1
    L[:, degenerate] = 0
    L[degenerate, :] = 0
    return L


def computeLaplaceMatrix2(dMatrix, t, logeps=mp.mpfr("-10")):
    """
    Compute heat approximation to Laplacian matrix using logarithms.

    Use gmpy2 mpfr to gain more precision.

    This is slow, but more accurate.

    Cutoff for really small values, and row elimination if degenerate.
    """
    lt = np.log(2 / t)
    L = np.sqrt(dMatrix * dMatrix)
    L /= -2.0 * t
    # numpy floating point errors likely below
    logdensity = sm.logsumexp(L, axis=1)
    # logdensity = logaddexp.reduce(np.sort(L, axis=1), axis=1)
    # compute log(density-1):
    # np.log(np.expm1(logdensity))
    # logdensity + np.log1p(-exp(logdensity))

    # sum in rows must be 1
    L = np.exp(L - logdensity[:, None] + lt)
    # fix diagonal to account for -f(x)?
    # L_t matrix is the unajusted one - scaled identity
    L[np.diag_indices(len(L))] -= 2.0 / t
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
        Ric[i, :] = cdc[i] / 2.0
    return Ric


def CDC1(L, f, g):
    """ Compute carre-du-champ for L. """
    u = f * g
    cdc = L.dot(u) - f * L.dot(g) - g * L.dot(f)
    return cdc / 2


# for test purposes.   Tested - and it at least for simple examples the
# Ricci's are all the same.
def coarseRicciold(L, dMatrix):
    Ric = np.zeros((len(dMatrix), len(dMatrix)))
    for i in range(len(L)):
        for j in range(len(L)):
            # pay close attention to this if it becomes asymmetric
            f_ij = dMatrix[:, i] - dMatrix[:, j]
            cdc1 = CDC1(L, f_ij, f_ij)
            cdc2 = L.dot(cdc1) - 2 * CDC1(L, f_ij, L.dot(f_ij))
            Ric[i, j] = cdc2[i] / 2

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


# This sections provides some simple test sets

# k,l are sizes of points.  sigma is about how far away points in the same
# cluster are.  Sigma must be positive
def onedimensionpair(k, l, sigma):
    X = np.random.normal(size=(k, 1))
    Y = np.random.normal(size=(l, 1)) + 2 / sigma
    Z = np.concatenate((X, Y))
    # print X
    # print Y
    print Z
    n = len(Z)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = (Z[i] - Z[j]) * (Z[i] - Z[j])
    dist = sigma * dist
    return dist


# returns distance squared for cyclical graph with n points, with noise added
def cyclegraph(n, noise):
    dist = np.zeros((n, n))
    ndist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.amin([(i - j) % n, (j - i) % n])
            ndist[i, j] = dist[i, j] * noise * np.random.randn(1)
    dist = dist * dist
    dist = dist + ndist + ndist.transpose()
    return dist


# returns distance squared.  Object is a pair of simplices with distance
# separation from each other, and internal distance 1.  Add some noise.
def closefarsimplices(n, noise, separation):
    dist = np.zeros((2 * n, 2 * n))
    symmetric(dist, noise)  # This isn't quite the object we want FIXME
    return dist


def metricize(dist):  # Only minimizes over two-stop paths not all
    dist = np.sqrt(dist)
    olddist = dist + 1
    d_ij = dist
    different = (olddist == dist).all()
    while(not different):
        # rint 'in loop'
        olddist = dist
        for i in range(len(dist)):
            for j in range(len(dist)):
                for k in range(len(dist)):
                    dijk = dist[i, k] + dist[k, j]
                    d_ij[i, j] = np.amin([d_ij[i, j], dijk])
                dist[i, j] = d_ij[i, j]
        different = (olddist == dist).all()
    return dist ** 2


dist = onedimensionpair(2, 3, noise)
dist = cyclegraph(6, noise)
#dist = closefarsimplices(n, 0.1, 1)


dist = metricize(dist)
L = computeLaplaceMatrix2(dist, t)

Ricci = coarseRicci3(L, dist)


print 'initial distance'
print dist
print 'initial Ricci'
print Ricci


ne.evaluate("dist-eta*Ricci", out=dist)

initial_L1 = dist.sum()

for i in range(runs + show + 3):
    L = computeLaplaceMatrix2(dist, t)
    Ricci = coarseRicci3(L, dist)
    ne.evaluate("dist-eta*Ricci", out=dist)
    dist = ne.evaluate("(dist + distT)/2",
                       global_dict={'distT': dist.transpose()})

    # total_distance = dist.sum()
    # dist = (total_distance0/total_distance)*dist
    nonzero = dist[np.nonzero(dist)]
    mindist = np.amin(nonzero)
    s1 = mindist
    s2 = dist.sum()
    # print t
    #ne.evaluate("dist/s", out=dist)

    dist = np.clip(dist, 0, CLIP)
    if rescale == 'L1':
        ne.evaluate("initial_L1*dist/s2", out=dist)
    if rescale == 'min':
        ne.evaluate("dist/s1", out=dist)
    dist = metricize(dist)
    if i % show == 2:
    # print Ricci
        print "dist for ", i, "  time"
        print dist
        print 't = ', t
        # print Ricci
        # print Ricci/dist, '<- Ricc/dist'
        print '---------'
