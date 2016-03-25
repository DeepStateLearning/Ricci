""" Other helpful functions. """

import timeit
import numpy as np
import numexpr as ne
from numba import jit
import threading


def test_speed(f, *args, **kwargs):
    """ Test the speed of a function. """
    if 'repeat' in kwargs:
        repeat = kwargs['repeat']
    else:
        repeat = 5
    t = timeit.repeat(lambda: f(*args), repeat=repeat, number=1)
    print 'Fastest out of 5: {} s'.format(min(t))


def metricize3(dist):
    """
    Metricize a matrix of "squared distances".

    Modifies the array in-place.

    Only minimizes over two-stop paths not all.
    """
    ne.evaluate("sqrt(dist)", out=dist)
    error = 1
    new = np.zeros_like(dist)
    while error > 1E-12:
        error = 0.0
        for i in xrange(len(dist)):
            new[i, :] = np.amin(dist[i, :, None] + dist, axis=0)
            np.minimum(dist[i, :], new[i, :], out=new[i, :])
            error += np.sum(dist[i, :] - new[i, :])
        dist[:, :] = new
    ne.evaluate('dist**2', out=dist)


def parallel(num, numthreads=8):
    """
    Decorator to execute a function with the first few args split into chunks.

    All arguments should be numpy arrays.

    The chunks should be independent computationally.

    Can be used as a replacement for OpenMP parallel for applied to outer loop.

    Arguments:
        num        - number of function arguments to split
        numthreads - number of threads

    One of the arguments should be the output array. This one should be split
    together with at least one of the input arguments.

    Example:
        (n,k), (k,l) -> (n, l) : Put output as first argument, and choose to
                    split 2 arguments. Then the problem will be split into
                    numthreads subproblems with n replaced by n/numthreads.

    Notes:
        Function relies on array_split returning views so that each chunk
        is placed in the appropriate part of the output array automatically.

        Chunks will run parallel only if GIL is released by the function.
        E.g. use nopython and nogil mode in numba jit.
    """
    def chunks_decorator(fun):
        def func_wrapper(*args):
            # FIXME switch to Queue/workers setup for smaller chunks ?
            chunks = [np.array_split(arg, numthreads) for arg in args[:num]]
            chunks = map(list, zip(*chunks))
            for chunk in chunks:
                chunk.extend(args[num:])
            threads = [threading.Thread(target=fun, args=chunk)
                       for chunk in chunks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        return func_wrapper
    return chunks_decorator


@parallel(2, numthreads=4)
@jit("void(f8[:,:], f8[:,:], f8[:,:])", nopython=True, nogil=True)
def _inner_loops(dist, new, distunsplit):
    for i in xrange(len(dist)):
        for j in xrange(len(distunsplit)):
            d_ij = dist[i, j]
            for k in xrange(len(distunsplit)):
                dijk = dist[i, k] + distunsplit[k, j]
                if dijk < d_ij:
                    d_ij = dijk
            new[i, j] = d_ij


def metricize2(dist):
    """
    Metricize a matrix of "squared distances".

    Only minimizes over two-stop paths not all.
    """
    ne.evaluate("sqrt(dist)", out=dist)
    error = 1
    new = np.zeros_like(dist)
    while error > 1E-12:
        _inner_loops(dist, new, dist)
        error = ne.evaluate("sum(dist-new)")
        dist[:, :] = new
    ne.evaluate('dist**2', out=dist)


metricize = metricize3


def sanitize(sqdist,  how, clip=np.inf, norm=None):
    """
    Clean up the distance matrix.

    Clip large values, metricize and renormalize.
    """
    np.clip(sqdist, 0, clip, out=sqdist)
    metricize(sqdist)
    if how =='L1' :
        try:
            float(norm)
            s2 = sqdist.sum()
            ne.evaluate("norm*sqdist/s2", out=sqdist)
        except:
            if norm == 'min':
                nonzero = sqdist[np.nonzero(sqdist)]
                mindist = np.amin(nonzero)
                ne.evaluate("sqdist/mindist", out=sqdist)
    if how == 'L_inf' :
        try:
            float(norm)
            s2 = sqdist.max()
            ne.evaluate("norm*sqdist/s2", out=sqdist)
        except:
            if norm == 'min':
                nonzero = sqdist[np.nonzero(sqdist)]
                mindist = np.amin(nonzero)
                ne.evaluate("sqdist/mindist", out=sqdist)
        


def is_metric(sqdist, eps=1E-12):
    """ Check if the matrix is a true squared distance matrix. """
    dist = ne.evaluate("sqrt(sqdist)")
    for i in xrange(len(dist)):
        temp = ne.evaluate("diT + dist - di < -eps",
                           global_dict={'diT': dist[i, :, None],
                                        'di': dist[i, :]})
        if np.any(temp):
            return False
    return True

    
def is_stuck(a,b,eta):  # Check if the ricci flow is stuck
    c = np.absolute(a-b)
    if c.max()<eta/10:
        return True
    else :return False
    
 
def is_clustered(sqdist, threshold):
    """
    Check if the metric is cluster.

    If the relations d(x,y)<threshold partitions the point set, returns True.
    """
    n = len(sqdist)
    partition = (sqdist < threshold)
    print partition
    for i in range(n):
        # setpart = partition[i, :]
        for j in range(i, n):
            if (partition[i, :] * partition[j, :]).any():
                if not np.array_equal(partition[i, :], partition[j, :]):
                    return False
    print 'clustered!!'
    #np.savetxt("clust.csv", partition, fmt="%5i", delimiter=",")
    #print 'saved to cust.csv'
    return True


def color_clusters(sqdist, threshold):
    """Assuming the metric is clustered, return a colored array."""
    n = len(sqdist)
    partition = (sqdist < threshold)
    clust = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if partition[i, j]:
                clust[i] = j
                break
    return clust


import unittest


class ToolsTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def test_correct(self):
        """ Test correctness of metricize2 on random data sets. """
        threshold = 1E-10
        print
        for n in range(200, 500, 100):
            d = np.random.rand(n, n)
            d += d.T
            np.fill_diagonal(d, 0)
            d2 = d.copy()
            d3 = d.copy()
            metricize2(d)
            metricize3(d2)
            print "Changed entries: {} out of {}." \
                .format(n*n - np.isclose(d, d3).sum(), n*n)
            error = np.max(np.abs(d-d2))
            print "Absolute difference between methods: ", error
            self.assertLess(error, threshold)
            self.assertTrue(is_metric(d))

    def speed(self, f):
        """ Test speed on larger data sets. """
        print
        s = 200
        for n in range(s, 4*s, s):
            d = np.random.rand(n, n)
            d += d.T
            np.fill_diagonal(d, 0)
            test_speed(f, d, repeat=1)

    def test_speed_metricize2(self):
        """ Speed of the parallelized metricize. """
        self.speed(metricize2)

    def test_speed_metricize3(self):
        """ Speed of the numpy metricize. """
        self.speed(metricize3)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ToolsTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
