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
    print 'Fastest out of {}: {} s'.format(repeat, min(t))


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
            dist[i, :] = new[i, :]
        print error
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


def build_fastmath_extension():
    """
    Build fastmath C/C++ extension.

    Requires fast-math, which impacts accuracy, but can be very fast.

    Metricize from this module is much faster than fully optimized SSE BLIS
    on processors with AVX.
    """
    from scipy.weave import ext_tools
    mod = ext_tools.ext_module('ctools_fastmath')
    # type declarations
    d = np.zeros((2, 2))
    d2 = np.zeros((2, 2))

    # metricize via BLIS framework
    with open('cpp/dgemm.c', 'r') as f:
        support_code = f.read()
    with open('cpp/metricize_dgemm.c', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize_gemm', code, ['d', 'd2'])
    func.customize.add_support_code(support_code)
    mod.add_function(func)
    # mod.customize.add_header('<omp.h>')
    mod.customize.add_header('<cmath>')
    mod.customize.add_header('<x86intrin.h>')
    mod.compile(extra_compile_args=["-O3 -fopenmp", "-march=native",
                                    "-fomit-frame-pointer", "-ffast-math",
                                    "-mfpmath=sse"],
                verbose=2, libraries=['gomp'],
                )


def build_extension():
    """
    Build C/C++ extension module ctools.

    Contains:
        - metricize with fully parallelized triangular loop
        - somewhat random metricize
        - connected components
        - clustered check
    """
    from scipy.weave import ext_tools
    mod = ext_tools.ext_module('ctools')
    # type declarations
    d = np.zeros((2, 2))
    d2 = np.zeros((2, 2))
    threshold = 0.5
    colors = np.zeros(2, dtype=int)

    # metricize using fully parallelized triangular loop
    # this function is actually slower with fastmath (when avx2 is used? a bug?)
    with open('cpp/metricize.cpp', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize', code, ['d'])
    mod.add_function(func)

    # metricize using fully parallelized triangular loop
    with open('cpp/metricize_random.cpp', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize_random', code, ['d'])
    mod.add_function(func)

    # find connected components and number them
    # fills last argument with component numbers for vertices
    # returns number of components
    with open('cpp/components.cpp', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('components', code,
                                  ['d', 'threshold', 'colors'])
    mod.add_function(func)

    # check if clustered function
    with open('cpp/clustered.cpp', 'r') as f:
        support_code = f.read()
    func = ext_tools.ext_function('clustered', """
    return_val = clustered_depthfirst(Nd[0], threshold, d);
    """, ['d', 'threshold'])
    func.customize.add_support_code(support_code)
    mod.add_function(func)

    # metricize via BLIS framework
    # FIXME add avx kernel
    with open('cpp/dgemm_asm_sse.c', 'r') as f:
        support_code = f.read()
    with open('cpp/metricize_dgemm.c', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize_gemm', code, ['d', 'd2'])
    func.customize.add_support_code(support_code)
    mod.add_function(func)
    # mod.customize.add_header('<omp.h>')
    mod.customize.add_header('<vector>')
    mod.customize.add_header('<cmath>')
    mod.customize.add_header('<x86intrin.h>')
    mod.compile(extra_compile_args=["-O3 -fopenmp", "-march=native",
                                    "-fomit-frame-pointer",
                                    "-mfpmath=sse"],
                verbose=2, libraries=['gomp'],
                )


# set current metricize method
metricize = metricize3
# replace wih C++ if possible
try:
    build_extension()
    build_fastmath_extension()
    import ctools
    import ctools_fastmath

    def metricize4(dist):
        """
        Metricize a matrix of "squared distances".

        C++ extension leveraging matrix symmetry and OpenMP.
        """
        ctools.metricize(dist)

    def metricize4b(dist):
        """
        Metricize a matrix of "squared distances".

        C++ extension leveraging matrix symmetry and OpenMP.
        """
        ctools.metricize_random(dist)

    def metricize5(dist, temp):
        """
        Metricize based on BLIS framework for BLAS.

        Modified ulmBLAS code for dgemm_nn.
        """
        ctools.metricize_gemm(dist, temp)

    def metricize5b(dist, temp):
        """
        Metricize based on BLIS framework for BLAS.

        Modified ulmBLAS code for dgemm_nn.
        """
        ctools_fastmath.metricize_gemm(dist, temp)

    def components(dist, threshold, colors):
        """
        Find connected components based on closeness threshold.

        Returns number of components and fills colors array with
        numbers (colors) for vertices.
        """
        return ctools.components(dist, threshold, colors)

    metricize = metricize5b
except:
    print "   !!! Error: C++ extension failed to build !!!   "
    metricize4 = metricize


def sanitize(sqdist, temp, how='L_inf', clip=np.inf, norm=1.0):
    """
    Clean up the distance matrix.

    Clip large values, metricize and renormalize.
    """
    # pylama:ignore=W0612
    np.clip(sqdist, 0.0, clip, out=sqdist)
    metricize(sqdist, temp)
    # metricize4b(sqdist)
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


def is_clustered_old(sqdist, threshold):
    """
    Check if the metric is cluster.

    If the relations d(x,y)<threshold partitions the point set, return True.
    """
    n = len(sqdist)
    partition = (sqdist < threshold)
    # print partition
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

try:
    ctools.clustered

    def is_clustered(sqdist, threshold):
        """
        Check if the metric is cluster.

        If the relations d(x,y)<threshold partitions the point set, return True.

        Implemented in C++ using depth first connected component search.
        """
        return ctools.clustered(sqdist, threshold)
except:
    is_clustered = is_clustered_old


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
    clust_set = set(clust)
    map = np.zeros(n, dtype=int)
    count = 0
    for i in range(n):
        map[i] = count
        if i in clust_set:
            count += 1

    for i in range(n):
        clust[i] = map[clust[i]]

    return clust


import unittest


class ToolsTests (unittest.TestCase):

    """ Correctness and speed tests. """

    def correct(self, f):
        """ Test correctness against metricize3 on random data sets. """
        threshold = 1E-10
        print
        for n in range(200, 500, 100):
            d = np.random.rand(n, n)
            d = d + d.T
            np.fill_diagonal(d, 0)
            d2 = d.copy()
            d3 = d.copy()
            metricize3(d)
            try:
                f(d2)
            except:
                temp = d.copy()
                f(d2, temp)
            print "Changed entries: {} out of {}." \
                .format(n*n - np.isclose(d, d3).sum(), n*n)
            error = np.max(np.abs(d-d2))
            print "Absolute difference between methods: ", error
            self.assertLess(error, threshold)
            self.assertTrue(is_metric(d))

    def test_metricize2(self):
        """ Test metricize2 (parallelized) against metricize3 (numpy). """
        self.correct(metricize2)

    def test_metricize4(self):
        """ Test metricize4 (C++) against metricize3 (numpy). """
        self.correct(metricize4)

    def test_metricize5(self):
        """ Test metricize5 (C, asm, and BLIS) against metricize3 (numpy). """
        self.correct(metricize5)

    def test_metricize5b(self):
        """ Test metricize5b (pure C and BLIS) against metricize3 (numpy). """
        self.correct(metricize5)

    def speed(self, f, s=200):
        """ Test speed on larger data sets. """
        print
        for n in range(s, 4*s, s):
            print "Points: ", n
            d = np.random.rand(n, n)
            d = d + d.T
            np.fill_diagonal(d, 0)
            if f in (metricize5, metricize5b):
                temp = d.copy()
                test_speed(f, d, temp, repeat=1)
            else:
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
        self.speed(metricize4, 500)

    def test_speed_metricize4b(self):
        """ Speed of the random C++ metricize. """
        self.speed(metricize4b)

    def test_speed_metricize5(self):
        """ Speed of the C and asm BLIS metricize. """
        self.speed(metricize5)
        self.speed(metricize5, 500)
        A = np.random.rand(1500, 1500)
        print "Same size np.dot(A, A) (metricize does a few multiplications):"
        test_speed(np.dot, A, A, repeat=1)

    def test_speed_metricize5b(self):
        """ Speed of the pure C BLIS metricize. """
        self.speed(metricize5b)
        self.speed(metricize5b, 500)
        A = np.random.rand(1500, 1500)
        print "Same size np.dot(A, A) (metricize does a few multiplications):"
        test_speed(np.dot, A, A, repeat=1)

    def test_clustered(self):
        """ Check if fast clustered check works. """
        from ctools import clustered
        for n in range(100, 120):
            # 5 clusters
            A = np.random.randint(5, size=n)+1
            A = np.array(A, dtype=float)
            AA = ne.evaluate("AT*A-A*A", global_dict={'AT': A[:, None]})
            ne.evaluate("where(AA!=0, 1.0, AA)", out=AA)
            # test clustered first
            self.assertTrue(is_clustered(AA, 0.5))
            for k in range(0, 10):
                # introduce a few extra connections
                i, j = np.random.randint(n, size=2)
                AA[i, j] = AA[j, i] = 0.0
                self.assertTrue(is_clustered_old(AA, 0.5) == clustered(AA, 0.5))


if __name__ == "__main__":
    # FIXME add missing tests for components
    suite = unittest.TestLoader().loadTestsFromTestCase(ToolsTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
