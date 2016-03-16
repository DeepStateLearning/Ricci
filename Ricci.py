""" Coarse Ricci matrix. """

import numpy as np
import numexpr as ne

# TODO
#   add tests
#   single threaded Ricci row computation parallelized with respect to rows


def coarseRicci2(L, sqdist):
    """ numexpr parallelized Ricci. """
    Ric = np.zeros_like(sqdist)
    for i in xrange(len(L)):
        # now f_i and cdc are matrices
        f_i = ne.evaluate("ci-sqdist",
                          global_dict={'ci': sqdist[:, i, None]})
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


def coarseRicciold(L, sqdist):
    """ Slow but surely correct Ricci computation. """
    Ric = np.zeros((len(sqdist), len(sqdist)))
    for i in range(len(L)):
        for j in range(len(L)):
            # pay close attention to this if it becomes asymmetric
            f_ij = sqdist[:, i] - sqdist[:, j]
            cdc1 = CDC1(L, f_ij, f_ij)
            cdc2 = L.dot(cdc1) - 2 * CDC1(L, f_ij, L.dot(f_ij))
            Ric[i, j] = cdc2[i] / 2

    return Ric


def coarseRicci3(L, sqdist):
    """
    Precompute Ld first and try to avoid mat-mat multiplications.

    This one is about 3x faster, but requires a bit more memory.
    """
    Ric = np.zeros_like(sqdist)
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
    return Ric


# currently best method
coarseRicci = coarseRicci3
