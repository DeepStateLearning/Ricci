""" Other helpful functions. """

import timeit
import numpy as np
import numexpr as ne
from numba import jit


def test(f, args_string):
    """ Test the speed of a function. """
    print f.__name__
    t = timeit.repeat("%s(%s)" % (f.__name__, args_string),
                      repeat=5, number=1,
                      setup="from __main__ import %s, %s" % (f.__name__,
                                                             args_string))
    print min(t)


def metricize(dist):
    """
    Metricize a matrix of "squared distances".

    Only minimizes over two-stop paths not all.
    """
    ne.evaluate("sqrt(dist)", out=dist)
    error = 1
    while error > 1E-12:
        # thid does not work
        # olddist = dist
        # olddist is a reference to dist
        error = 0.0
        for i in xrange(len(dist)):
                d_i = np.amin(dist[i, :, None] + dist, axis=1)
                error += np.sum(dist[i, :] - d_i)
                dist[i, :] = d_i
    ne.evaluate('dist**2', out=dist)


@jit("void(f8[:,:])", nopython=True, nogil=True)
def metricize2(dist):
    """
    Metricize a matrix of "squared distances".

    Only minimizes over two-stop paths not all.
    """
    dist = np.sqrt(dist)
    error = 1
    while error > 1E-12:
        error = 0
        for i in xrange(len(dist)):
            for j in xrange(len(dist)):
                old = d_ij = dist[i, j]
                for k in xrange(len(dist)):
                    dijk = dist[i, k] + dist[k, j]
                    if dijk < d_ij:
                        d_ij = dijk
                dist[i, j] = d_ij
                error += old-d_ij
    dist *= dist

if __name__ == "__main__":
    import data
    dist = data.closefarsimplices(100, 0.1, 5)
    zeros = np.zeros_like(dist)
    test(metricize, "dist")
    test(metricize2, "dist")
