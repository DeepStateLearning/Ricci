""" Approximate Laplace matrix via heat kernel. """

import numpy as np
import gmpy2 as mp
import scipy.misc as sm

# gmpy2 setup for numpy object arrays
mp.get_context().precision = 200
exp = np.frompyfunc(mp.exp, 1, 1)
expm1 = np.frompyfunc(mp.expm1, 1, 1)
log = np.frompyfunc(mp.log, 1, 1)
is_finite = np.frompyfunc(mp.is_finite, 1, 1)
to_mpfr = np.frompyfunc(mp.mpfr, 1, 1)
to_double = np.frompyfunc(float, 1, 1)


def logsumexp(a):
    """ mpfr compatible minimal logsumexp version. """
    m = np.max(a, axis=1)
    return log(np.sum(exp(a - m[:, None]), axis=1)) + m


def computeLaplaceMatrix(sqdist, t, logeps=mp.mpfr("-10")):
    """
    Compute heat approximation to Laplacian matrix using logarithms and gmpy2.

    Use mpfr to gain more precision.

    This is slow, but more accurate.

    Cutoff for really small values, and row/column elimination if degenerate.
    """
    # cutoff ufunc
    cutoff = np.frompyfunc((lambda x: mp.inf(-1) if x < logeps else x), 1, 1)

    t2 = mp.mpfr(t)
    lt = mp.log(2 / t2)
    d = to_mpfr(sqdist)
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


def computeLaplaceMatrix2(sqdist, t):
    """
    Compute heat approximation to Laplacian matrix using logarithms.

    This is faster, but not as accurate.
    """
    lt = np.log(2 / t)
    L = sqdist / (-2.0 * t)  # copy of sqdist is needed here anyway
    # numpy floating point errors likely below
    logdensity = sm.logsumexp(L, axis=1)
    # sum in rows must be 1, except for 2/t factor
    L = np.exp(L - logdensity[:, None] + lt)
    # fix diagonal to account for -f(x)?
    L[np.diag_indices(len(L))] -= 2.0 / t
    return L


# currently best method
Laplacian = computeLaplaceMatrix2
