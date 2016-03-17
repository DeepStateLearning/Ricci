""" Approximate Laplace matrix via heat kernel. """

import numpy as np
import scipy.misc as sm


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

try:
    # gmpy2 setup for numpy object arrays
    import gmpy2 as mp
    mp.get_context().precision = 100
    _exp = np.frompyfunc(mp.exp, 1, 1)
    _expm1 = np.frompyfunc(mp.expm1, 1, 1)
    _log = np.frompyfunc(mp.log, 1, 1)
    _is_finite = np.frompyfunc(mp.is_finite, 1, 1)
    _to_mpfr = np.frompyfunc(mp.mpfr, 1, 1)
    _to_double = np.frompyfunc(float, 1, 1)
    _cutoff = np.frompyfunc((lambda x, le: mp.inf(-1) if x < le else x), 2, 1)

    def _logsumexp(a):
        """ mpfr compatible minimal logsumexp version. """
        m = np.max(a, axis=1)
        return _log(np.sum(_exp(a - m[:, None]), axis=1)) + m

    def computeLaplaceMatrix(sqdist, t, logeps=mp.mpfr("-10")):
        """
        Compute heat approximation to Laplacian using logarithms and gmpy2.

        Use mpfr to gain more precision.

        This is slow, but more accurate.

        Cutoff for really small values based on logeps.
        Row/column elimination if degenerate.
        """
        # cutoff ufunc

        t2 = mp.mpfr(t)
        lt = mp.log(2 / t2)
        d = _to_mpfr(sqdist)
        L = d * d
        L /= -2 * t2
        _cutoff(L, out=L)
        logdensity = _logsumexp(L)
        L = _exp(L - logdensity[:, None] + lt)
        L[np.diag_indices(len(L))] -= 2 / t2
        L = np.array(_to_double(L), dtype=float)
        # if just one nonzero element, then erase row and column
        degenerate = np.sum(L != 0.0, axis=1) <= 1
        L[:, degenerate] = 0
        L[degenerate, :] = 0
        return L
except:
    print "Warning: gmpy2 is not available. "

    def computeLaplaceMatrix(sqdist, t, logeps):
        """ No gmpy2, so just run a different implementation. """
        return computeLaplaceMatrix2(sqdist, t)



# currently best method
Laplacian = computeLaplaceMatrix2
