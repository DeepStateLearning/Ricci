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
    mp.get_context().precision = 200
    _exp = np.frompyfunc(mp.exp, 1, 1)
    _log = np.frompyfunc(mp.log, 1, 1)
    _is_finite = np.frompyfunc(mp.is_finite, 1, 1)
    _to_mpfr = np.frompyfunc(mp.mpfr, 1, 1)
    _to_double = np.frompyfunc(float, 1, 1)
    _cutoff = np.frompyfunc((lambda x, le: mp.inf(-1) if x < le else x), 2, 1)
    WITH_GMPY2 = True

    def _logsumexp(a):
        """ mpfr compatible minimal logsumexp version. """
        m = np.max(a, axis=1)
        return _log(np.sum(_exp(a - m[:, None]), axis=1)) + m

    def computeLaplaceMatrix(sqdist, t, logeps=mp.mpfr("-20")):
        """
        Compute heat approximation to Laplacian using logarithms and gmpy2.

        Use mpfr to gain more precision.

        This is slow, but more accurate.

        Cutoff for really small values based on logeps.
        Row/column elimination if degenerate.
        """
        t2 = mp.mpfr(t)
        lt = mp.log(2 / t2)
        L = _to_mpfr(sqdist) / (-2 * t2)
        _cutoff(L, logeps, out=L)
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
    WITH_GMPY2 = False

    def computeLaplaceMatrix(sqdist, t, logeps=None):
        """ No gmpy2, so just run a different implementation. """
        return computeLaplaceMatrix2(sqdist, t)


# currently best method
Laplacian = computeLaplaceMatrix2

#
# tests based on computeLaplaceMatrix2
#
import unittest


class LaplaceTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def test_small(self):
        """ Test correctness on small data sets. """
        self.assertTrue(WITH_GMPY2, "gmpy2 not available !")
        import data
        threshold = 1E-10
        print
        for d in data.tests('small'):
            # compare without cutoff
            L1 = computeLaplaceMatrix(d, 0.1, logeps=mp.inf(-1))
            L2 = computeLaplaceMatrix2(d, 0.1)
            error = np.max(np.abs(L1-L2))
            print "Absolute error: ", error
            self.assertLess(error, threshold)

    def row_sums(self, f):
        """ Test if rows add up to 0. """
        import data
        threshold = 1E-10
        print
        for d in data.tests('small'):
            error = np.max(np.abs(np.sum(f(d, 0.1), axis=1)))
            print "Absolute error: ", error
            self.assertLess(error, threshold)

    def test_rows_numpy(self):
        """ Check sums of rows with numpy. """
        self.row_sums(computeLaplaceMatrix2)

    def test_rows_gmpy2(self):
        """ Check sums of rows with gmpy2. """
        self.assertTrue(WITH_GMPY2, "gmpy2 not available !")
        self.row_sums(computeLaplaceMatrix)

    def speed(self, f):
        """ Test speed on larger data sets. """
        import data
        from tools import test_speed
        d = data.closefarsimplices(200, 0.1, 5)[0]
        print "\nPoints: 200"
        test_speed(f, d, 0.1)
        d = data.closefarsimplices(400, 0.1, 5)[0]
        print "Points: 400"
        test_speed(f, d, 0.1)

    def test_speed_numpy(self):
        """ Speed with numpy. """
        self.speed(computeLaplaceMatrix2)

    def test_speed_gmpy2(self):
        """ Speed with gmpy2. """
        self.assertTrue(WITH_GMPY2, "gmpy2 not available !")
        self.speed(computeLaplaceMatrix)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(LaplaceTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
