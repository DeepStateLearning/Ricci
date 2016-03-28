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

    Pure numpy implementation.
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


def parallel(num, numthreads=4):
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
                dijk = dist[i, k] + distunsplit[j, k]
                if dijk < d_ij:
                    d_ij = dijk
            new[i, j] = d_ij


def metricize2(dist):
    """
    Metricize a matrix of "squared distances".

    Compiled using numba JIT and parallelized.
    """
    ne.evaluate("sqrt(dist)", out=dist)
    error = 1
    new = np.zeros_like(dist)
    while error > 1E-12:
        _inner_loops(dist, new, dist)
        error = ne.evaluate("sum(dist-new)")
        dist[:, :] = new
    ne.evaluate('dist**2', out=dist)


# metricize2 runs slowly the first time
metricize2(np.zeros((16, 16)))


def build_extension():
    """ Build C++ extension with metricize. """
    from scipy.weave import ext_tools
    mod = ext_tools.ext_module('ctools')
    # type declarations
    d = np.zeros((2, 2))
    code = r"""
    double error=1;
    const int n = Nd[0];
    const int r = n*(n-1)/2;
    const int w = n%2 ? (n-1)/2 : (n-1);
    std::vector<double> tril;
    tril.resize(r);

    // sqdist to dist
#pragma omp parallel for
    for (int i=0; i<n*n; i++)
        d[i] = sqrt(d[i]);

    while (error > 10e-12)
    {
        error = 0;
#pragma omp parallel shared(error)
{
        double d_ij, dijk, old;
#pragma omp for reduction(+:error) schedule(static)
        for (int l=0; l<r; ++l)
        {
            // lower triangle from linear index to improve parallel loop
            int i = l/w;
            int j = l%w;
            if (j>=i)
            {
                i = n-1-i;
                j = n-2-j;
            }
            old = d_ij = d[i*n+j];
            for (int k=0; k<n; ++k)
            {
                dijk = d[i*n+k] + d[j*n+k];
                if (dijk < d_ij)
                    d_ij = dijk;
            }
            if (old>d_ij)
            {
                error += old-d_ij;
                tril[l] = d_ij;
            }
        }
#pragma omp for
        for (int l=0; l<r; ++l)
            if (tril[l]>0)
            {
                // lower triangle from linear index to improve parallel loop
                int i = l/w;
                int j = l%w;
                if (j>=i)
                {
                    i = n-1-i;
                    j = n-2-j;
                }
                d[i*n+j] = d[j*n+i] = tril[l];
                tril[l] = 0;
            }
}
    }

    // dist to sqdist
#pragma omp parallel for
    for (int i=0; i<n*n; i++)
        d[i] *= d[i];
    """
    func = ext_tools.ext_function('metricize_dist', code, ['d'])
    # mod.customize.add_header('<omp.h>')
    mod.customize.add_header('<vector>')
    mod.customize.add_header('<cmath>')
    mod.add_function(func)
    mod.compile(extra_compile_args=["-O3 -fopenmp"],
                verbose=2, libraries=['gomp'],
                library_dirs=['/usr/local/lib'],
                include_dirs=['/usr/local/include'])


# set current metricize method
metricize = metricize3

# replace wih C++ if possible
try:
    build_extension()
    import ctools

    def metricize4(dist):
        """
        Metricize a matrix of "squared distances".

        C++ extension leveraging matrix symmetry and OpenMP.
        """
        ctools.metricize_dist(dist)

    metricize = metricize4
except:
    print "   !!! Error: C++ extension failed to build !!!   "
    metricize4 = metricize


def sanitize(sqdist,  how='L_inf', clip=np.inf, norm=1.0):
    """
    Clean up the distance matrix.

    Clip large values, metricize and renormalize.
    """
    # pylama:ignore=W0612
    np.clip(sqdist, 0, clip, out=sqdist)
    metricize(sqdist)
    try:
        norm = float(norm)
        assert norm > 0.0
    except:
        norm = 1.0
    if how == 'L1':
        s2 = sqdist.sum()
    else:  # how == 'L_inf' :
        s2 = sqdist.max()
    ne.evaluate("norm*sqdist/s2", out=sqdist)


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


def is_stuck(a, b, eta):
    """ Check if the ricci flow is stuck. """
    return ne.evaluate("a-b<eta/50").all()


def is_clustered(sqdist, threshold):
    """
    Check if the metric is cluster.

    If the relations d(x,y)<threshold partitions the point set, returns True.
    """
    n = len(sqdist)
    partition = (sqdist < threshold)
    #print partition
    for i in range(n):
        # setpart = partition[i, :]
        for j in range(i, n):
            if (partition[i, :] * partition[j, :]).any():
                if not np.array_equal(partition[i, :], partition[j, :]):
                    return False
    print 'clustered!!'
    # np.savetxt("clust.csv", partition, fmt="%5i", delimiter=",")
    # print 'saved to cust.csv'
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
            d = d + d.T
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
            d = d + d.T
            np.fill_diagonal(d, 0)
            test_speed(f, d, repeat=1)

    def test_speed_metricize2(self):
        """ Speed of the parallelized metricize. """
        self.speed(metricize2)

    def test_speed_metricize3(self):
        """ Speed of the numpy metricize. """
        self.speed(metricize3)

    def test_speed_metricize4(self):
        """ Speed of the C++ metricize. """
        self.speed(metricize4)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ToolsTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
