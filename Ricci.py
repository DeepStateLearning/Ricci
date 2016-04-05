""" Coarse Ricci matrix. """

import numpy as np
import numexpr as ne
import scipy.linalg as sl


def add_AB_to_C(A, B, C):
    """
    Compute C += AB in-place.

    This uses gemm from whatever BLAS is available.
    MKL requires Fortran ordered arrays to avoid copies.
    Hence we work with transpositions of default c-style arrays.
    """
    gemm = sl.get_blas_funcs("gemm", (A, B, C))
    assert np.isfortran(C.T) and np.isfortran(A.T) and np.isfortran(B.T)
    D = gemm(1.0, B.T, A.T, beta=1, c=C.T, overwrite_c=1)
    assert D.base is C


def applyRicci(sqdist, eta, T, Ricci, mode='sym'):
    """
    Apply coarse Ricci to a squared distance matrix.

    Can handle symmetric, max, and nonsymmetric modes.

    Note: eta can be a localizing kernel too.
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


# def coarseRicci2(L, sqdist):
#     """ numexpr parallelized Ricci. """
#     Ric = np.zeros_like(sqdist)
#     for i in xrange(len(L)):
#         # now f_i and cdc are matrices
#         f_i = ne.evaluate("ci-sqdist",
#                           global_dict={'ci': sqdist[:, i, None]})
#         # first CDC
#         cdc = L.dot(ne.evaluate("f_i*f_i"))
#         Lf = L.dot(f_i)
#         # end of CDC1 combined with CDC2
#         ne.evaluate("cdc/2.0-2.0*f_i*Lf", out=cdc)
#         cdc = L.dot(cdc)
#         ne.evaluate("cdc+f_i*LLf+Lf*Lf",
#                     global_dict={'LLf': L.dot(Lf)}, out=cdc)
#         Ric[i, :] = cdc[i] / 2.0
#     return Ric


def CDC1(L, f, g):
    """ Compute carre-du-champ for L. """
    u = f * g
    cdc = L.dot(u) - f * L.dot(g) - g * L.dot(f)
    return cdc / 2


def coarseRicciold(L, sqdist, Ric):
    """ Slow but surely correct Ricci computation. """
    for i in range(len(L)):
        for j in range(len(L)):
            # pay close attention to this if it becomes asymmetric
            f_ij = sqdist[:, i] - sqdist[:, j]
            cdc1 = CDC1(L, f_ij, f_ij)
            cdc2 = L.dot(cdc1) - 2 * CDC1(L, f_ij, L.dot(f_ij))
            Ric[i, j] = cdc2[i] / 2


def coarseRicci3(L, sqdist, Ric):
    """
    Precompute Ld first and try to avoid mat-mat multiplications.

    This one is about 3x faster, but requires a bit more memory.
    """
    Ld = L.dot(sqdist)
    for i in xrange(len(L)):
        # now f_i and cdc are matrices
        f_i = ne.evaluate("di-sqdist",
                          global_dict={'di': sqdist[:, i, None]})
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


def coarseRicci4(L, sqdist, R, temp1=None, temp2=None):
    """
    Fully optimized Ricci matrix computation.

    Requires 7 matrix multiplications and many entrywise operations.
    We use 3 temporary matrices, but we should only 2.

    Uses full gemm functionality to avoid creating intermediate matrices.

    Running time is O(n^3) as opposed to other implementations' O(n^4).

    R is the output array, while A and B are temporary matrices.
    """
    D = sqdist
    if temp1 is None:
        temp1 = np.zeros_like(sqdist)
    if temp2 is None:
        temp2 = np.zeros_like(sqdist)
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
    """
    Compute scalar curvature.

    Without creating any matrices.
    """
    density = ne.evaluate("sum(exp(-sqdist/t), axis=1)")
    # Scalar = np.diag(Ricci.dot(kernel))
    # same as
    Scalar = ne.evaluate("sum(Ricci*exp(-sqdist/t), axis=1)")
    # density = kernel.sum(axis=1)
    ne.evaluate("Scalar/density", out=Scalar)
    return Scalar

# currently best method
coarseRicci = coarseRicci4

#
# tests based on old Ricci
#
import unittest


class RicciTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def small(self, f, size='small', default=None):
        """ Test correctness on small data sets. """
        import data
        from Laplacian import Laplacian
        threshold = 1E-10
        if default is None:
            default = coarseRicciold
        print
        for d in data.tests(size):
            L = np.random.rand(*d.shape)
            R = np.random.rand(*d.shape)
            R2 = R.copy()
            Laplacian(d, 0.1, L)
            f(L, d, R)
            default(L, d, R2)
            error = np.amax(np.abs(R-R2))
            print "Absolute error: ", error
            self.assertLess(error, threshold)

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

    def test_Ricci3(self):
        """ Ricci3 compared to old mathematical Ricci. """
        self.small(coarseRicci3)

    def test_Ricci4(self):
        """ Ricci4 compared to old mathematical Ricci. """
        self.small(coarseRicci4)

    def test_Ricci4large(self):
        """ Ricci4 compared to Ricci3 on larger data. """
        self.small(coarseRicci4, size='large', default=coarseRicci3)

    def test_speed_Ricci3(self):
        """ Speed of Ricci3. """
        self.speed(coarseRicci3)

    def test_speed_Ricci4(self):
        """ Speed of Ricci4. """
        self.speed(coarseRicci4)
        self.speed(coarseRicci4, points=[500, 1000])

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(RicciTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
