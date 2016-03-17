""" Other helpful functions. """

import timeit
import numpy as np


def test_speed(f, *args):
    """ Test the speed of a function. """
    t = timeit.repeat(lambda: f(*args), repeat=5, number=1)
    print 'Fastest out of 5: {} s'.format(min(t))


def metricize(dist):
    """
    Metricize a matrix of "squared distances".

    Only minimizes over two-stop paths not all.
    """
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
