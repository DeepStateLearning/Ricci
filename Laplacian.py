""" Approximate Laplace matrix via heat kernel. """

import numexpr as ne
import numpy as np


def computeLaplaceMatrix(sqdist, t, L):
    """ Compute heat approximation to Laplacian matrix using logarithms. """
    # FIXME Is the scaling ok in here?
    lt = np.log(2.0 / t)
    ne.evaluate('sqdist / (-2.0 * t)', out=L)
    # sqdist is nonnegative, with 0 on the diagonal
    # so the largest element of each row of L is 0
    # no logsumexp needed
    density = ne.evaluate("sum(exp(L), axis=1)")[:, None]
    ne.evaluate("log(density)", out=density)
    # sum in rows must be 1, except for 2/t factor
    ne.evaluate('exp(L - density + lt)', out=L)
    # fix diagonal to account for -f(x)?
    L[np.diag_indices(len(L))] -= 2.0 / t

Laplacian = computeLaplaceMatrix

#
# tests based on computeLaplaceMatrix2
#
import unittest


class LaplaceTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def row_sums(self, f):
        """ Test if rows add up to 0. """
        import data
        threshold = 1E-10
        print
        for d in data.tests('small'):
            L = np.random.rand(*d.shape)
            f(d, 0.1, L)
            error = np.max(np.abs(np.sum(L, axis=1)))
            print "Absolute error: ", error
            self.assertLess(error, threshold)

    def test_rows_numpy(self):
        """ Check sums of rows with numpy. """
        self.row_sums(computeLaplaceMatrix)

    def speed(self, f):
        """ Test speed on larger data sets. """
        import data
        from tools import test_speed
        d = data.closefarsimplices(500, 0.1, 5)[0]
        print "\nPoints: 1000"
        L = np.random.rand(*d.shape)
        test_speed(f, d, 0.1, L)
        d = data.closefarsimplices(1000, 0.1, 5)[0]
        L = np.random.rand(*d.shape)
        print "Points: 2000"
        test_speed(f, d, 0.1, L)

    def test_speed_numpy(self):
        """ Speed with numpy. """
        self.speed(computeLaplaceMatrix)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(LaplaceTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
