""" Coarse Ricci matrix. """

import numpy as np
import numexpr as ne
import scipy.linalg as sl
from pyfftw import zeros_aligned


def add_AB_to_C(A, B, C):
    """
    Compute C += AB in-place.

    This uses gemm from whatever BLAS is available.
    MKL requires Fortran ordered arrays to avoid copies.
    Hence we work with transpositions of default c-style arrays.

    This function throws error if computation is not in-place.
    """
    gemm = sl.get_blas_funcs("gemm", (A, B, C))
    assert np.isfortran(C.T) and np.isfortran(A.T) and np.isfortran(B.T)
    D = gemm(1.0, B.T, A.T, beta=1, c=C.T, overwrite_c=1)
    assert D.base is C or D.base is C.base


def applyRicci(sqdist, eta, T, Ricci, mode='sym'):
    """
    Apply coarse Ricci to a squared distance matrix.

    Can handle symmetric, max, and nonsymmetric modes.

    Gaussian localizing kernel is used with T as variance parameter.
    """
    if 'sym' in mode:
        ne.evaluate('sqdist - (eta/2)*exp(-sqdist/T)*(Ricci+RicciT)',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'max' in mode:
        ne.evaluate(
            'sqdist - eta*exp(-sqdist/T)*where(Ricci<RicciT, RicciT, Ricci)',
            global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'dumb' in mode:
        ne.evaluate('sqdist*(1 - eta*exp(-sqdist/T))', out=sqdist)
    else:
        ne.evaluate('sqdist - eta*exp(-sqdist/T)*Ricci',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)


def coarseRicci(L, sqdist, R, temp1=None, temp2=None):
    """
    Fully optimized Ricci matrix computation.

    Requires 7 matrix multiplications and many entrywise operations.
    Only 2 temporary matrices are needed, and can be provided as arguments.

    Uses full gemm functionality to avoid creating intermediate matrices.

    R is the output array, while temp1 and temp2 are temporary matrices.
    """
    D = sqdist
    if temp1 is None:
        temp1 = zeros_aligned(sqdist.shape, n=32)
    if temp2 is None:
        temp2 = zeros_aligned(sqdist.shape, n=32)
    A = temp1
    B = temp2
    # this C should not exist
    B = ne.evaluate("D*D/4.0")
    L.dot(B, out=A)
    L.dot(D, out=B)
    ne.evaluate("A-D*B", out=A)
    L.dot(A, out=R)
    # the first two terms done
    L.dot(B, out=A)
    ne.evaluate("R+0.5*(D*A+B*B)", out=R)
    # Now R contains everything under overline
    ne.evaluate("R+dR-0.5*dA*D-dB*B",
                global_dict={'dA': np.diag(A).copy()[:, None],
                             'dB': np.diag(B).copy()[:, None],
                             'dR': np.diag(R).copy()[:, None]}, out=R)
    # Now R contains all but two matrix products from line 2
    L.dot(L, out=A)
    ne.evaluate("L*BT-0.5*A*D", global_dict={'BT': B.T}, out=A)
    add_AB_to_C(A, D, R)
    ne.evaluate("L*D", out=A)
    add_AB_to_C(A, B, R)
    # done!
    np.fill_diagonal(R, 0.0)


def getScalar(Ricci, sqdist, t):
    """ Compute scalar curvature. """
    density = ne.evaluate("sum(exp(-sqdist/t), axis=1)")
    # Scalar = np.diag(Ricci.dot(kernel))
    # same as
    Scalar = ne.evaluate("sum(Ricci*exp(-sqdist/t), axis=1)")
    # density = kernel.sum(axis=1)
    ne.evaluate("Scalar/density", out=Scalar)
    return Scalar

#
# tests based on old Ricci
#
import unittest


class RicciTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def speed(self, f, points=[100, 200]):
        """ Test speed on larger data sets. """
        import data
        from Laplacian import Laplacian
        from tools import test_speed
        for p in points:
            d = data.closefarsimplices(p, 0.1, 5)[0]
            print "\nPoints: {}".format(2*p)
            L = np.zeros_like(d)
            R = np.zeros_like(d)
            print "Laplacian: ",
            test_speed(Laplacian, d, 0.1, L)
            Laplacian(d, 0.1, L)
            print "Ricci: ",
            test_speed(f, L, d, R)

    def test_speed_Ricci(self):
        """ Speed of coarse Ricci compared to Laplacian. """
        self.speed(coarseRicci)
        self.speed(coarseRicci, points=[500, 1000])

if __name__ == "__main__":
    # FIXME any correctness tests?
    # FIXME add scalar curvature tests
    suite = unittest.TestLoader().loadTestsFromTestCase(RicciTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
