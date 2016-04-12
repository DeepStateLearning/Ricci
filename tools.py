""" Other helpful functions. """

import timeit
import numpy as np
import numexpr as ne
from pyfftw import zeros_aligned, simd_alignment
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(threshold=np.nan)


def test_speed(f, *args, **kwargs):
    """ Test the speed of a function. """
    if 'repeat' in kwargs:
        repeat = kwargs['repeat']
    else:
        repeat = 5
    t = timeit.repeat(lambda: f(*args), repeat=repeat, number=1)
    print 'Fastest out of {}: {} s'.format(repeat, min(t))


def get_matrices(M, n):
    """ Make sure M is aligned and generate n other matrices. """
    print "Aligning to ", simd_alignment, " bytes"
    m = zeros_aligned(M.shape)
    np.copyto(m, M)
    M = m
    lst = []
    for _ in xrange(n):
        lst.append(zeros_aligned(M.shape))
    lst.append(M)
    return lst


def init_plot(dim):
    """ Create empty plot container for later real time updates. """
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(1, 1, 1)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    return plt, ax


def graph(threshold, pointset, sqdist, ax, dim, mode="sort"):
    """
    Draw pointset as colored connected components.

    By default each component has a different color based on cluster number.

    They can also be colored using distance to one of the points of one of them.
    However, this does not work well for more than two clusters, since distances
    between clusters are about 1. (mode="dist")

    Components can also be sorted according to the distance to a point from one
    of them, to make less random looking coloring. (mode="sort")

    Distances are measured between representatives of the component, so close
    components may end up having large distance.
    """
    c = np.zeros(len(pointset), dtype=int)
    num = components(sqdist, 0.001, c)
    print "Number of components: ", num
    # now c contains uniquely numbered components

    # replace colors with distance to a point
    # component number and a representative
    values, points = np.unique(c, return_index=True)
    dists = sqdist[points[0], points]
    if mode == "dist":
        c = dists[c]
    elif mode == "sort":
        a = sorted(zip(values, dists), key=lambda e: e[1])
        a = np.array(a)[:, 0]
        c = a[c]
    if len(points) < 10:
        print "Distances between components: \n", sqdist[np.ix_(points, points)]
    else:
        print "Distances to component 0: \n", dists
    plt.cla()
    if dim == 2:
        ax.scatter(pointset[:, 0], pointset[:, 1],  c=c, cmap='gnuplot2')
        plt.axis('equal')
    elif dim == 3:
        # from mpl_toolkits.mplot3d import Axes3D
        ax.scatter(pointset[:, 0], pointset[:, 1], pointset[:, 2],  c=c,
                   cmap='gnuplot2')
    plt.draw()
    plt.pause(0.01)
    return num




def build_fastmath_extension():
    """
    Build fastmath C/C++ extension.

    Requires fast-math, which impacts accuracy, but can be very fast.

    Metricize from this module is much faster than fully optimized SSE BLIS
    on processors with AVX.
    """
    from scipy.weave import ext_tools
    mod = ext_tools.ext_module('ctools_fastmath')
    # number of physical cpus/core
    import mkl
    ncpus = mkl.get_max_threads()
    # type declarations
    d = np.zeros((2, 2))
    d2 = np.zeros((2, 2))
    limit = 4

    with open('cpp/fw_tiled.c', 'r') as f:
        support_code = f.read()
    with open('cpp/fw.c', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize_fw', code, ['d', 'd2'])
    func.customize.add_support_code(support_code)
    mod.add_function(func)

    # metricize via BLIS framework
    # with open('cpp/dgemm.c', 'r') as f:
    #     support_code = f.read()
    # with open('cpp/metricize_dgemm.c', 'r') as f:
    #     code = f.read()
    # func =ext_tools.ext_function('metricize_gemm', code, ['d', 'd2', 'limit'])
    # func.customize.add_support_code(support_code)
    # mod.add_function(func)
    mod.customize.add_header('<omp.h>')
    mod.customize.add_header('<cmath>')
    mod.customize.add_header('<x86intrin.h>')
    mod.compile(extra_compile_args=["-O3 -DNUMCORE={}".format(ncpus),
                                    "-fopenmp -march=native",
                                    "-fomit-frame-pointer", "-ffast-math",
                                    "-mfpmath=sse",
                                    "-Wno-unused-variable",
                                    "-ftree-vectorize"],
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
    # number of physical cpus
    import mkl
    ncpus = mkl.get_max_threads()
    print "Number of physical cores: ", ncpus
    # type declarations
    d = np.zeros((2, 2))
    d2 = np.zeros((2, 2))
    threshold = 0.5
    limit = 4
    colors = np.zeros(2, dtype=int)

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
    with open('cpp/dgemm.c', 'r') as f:
        support_code = f.read()
    with open('cpp/metricize_dgemm.c', 'r') as f:
        code = f.read()
    func = ext_tools.ext_function('metricize_gemm', code, ['d', 'd2', 'limit'])
    func.customize.add_support_code(support_code)
    mod.add_function(func)
    mod.customize.add_header('<omp.h>')
    mod.customize.add_header('<vector>')
    mod.customize.add_header('<cmath>')
    mod.customize.add_header('<x86intrin.h>')
    mod.compile(extra_compile_args=["-O3 -DNUMCORE={}".format(ncpus),
                                    "-fopenmp -march=native",
                                    "-fomit-frame-pointer",
                                    "-mfpmath=sse",
                                    "-Wno-unused-variable"],
                verbose=2, libraries=['gomp'],
                )


try:
    build_fastmath_extension()
    build_extension()
    import ctools
    import ctools_fastmath
except:
    # might have been built on a different machine
    # try rebuilding
    import os
    os.system("rm ctools*")
    build_fastmath_extension()
    build_extension()
    import ctools
    import ctools_fastmath


def metricize_fw(dist):
    """
    Metricize a matrix of "squared distances".

    Tiled Floyd-Warshall algorithm mixed with BLIS.

    !! Work in progress !!
     - only works for number of points divisible by 128
    """
    temp = dist.copy()
    ctools_fastmath.metricize_fw(dist, temp)


def metricize(dist, temp=None, limit=0):
    """
    Metricize based on BLIS framework for BLAS.

    Modified ulmBLAS code for dgemm_nn with optimal kernel.

    If limit is larger than 0, then only this many rounds will happen.
    """
    if temp is None:
        temp = zeros_aligned(dist.shape, n=32)
    # We use subtropical matrix multiplication since it is faster
    # Starting with Skylake the tropical one will be as fast
    ne.evaluate('exp(sqrt(dist))', out=dist)
    np.copyto(temp, dist)
    ctools.metricize_gemm(dist, temp, limit)
    ne.evaluate('log(dist)**2', out=dist)


def metricize_pureC(dist, temp=None, limit=0):
    """
    Metricize based on BLIS framework for BLAS.

    Modified ulmBLAS code for dgemm_nn.
    """
    if temp is None:
        temp = zeros_aligned(dist.shape, n=32)
    # We use subtropical matrix multiplication since it is faster
    # Starting with Skylake the tropical one will be as fast
    ne.evaluate('exp(sqrt(dist))', out=dist)
    np.copyto(temp, dist)
    ctools_fastmath.metricize_gemm(dist, temp, limit)
    ne.evaluate('log(dist)**2', out=dist)


def components(dist, threshold, colors):
    """
    Find connected components based on closeness threshold.

    Returns number of components and fills colors array with
    numbers (colors) for vertices.
    """
    return ctools.components(dist, threshold, colors)


def sanitize(sqdist, how='L_inf', clip=np.inf, norm=1.0, temp=None):
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
    # FIXME reimplement in C
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

    If the relations d(x,y)<threshold partitions the point set, return True.

    Implemented in C++ using modified depth first connected component search.
    """
    return ctools.clustered(sqdist, threshold)


def color_clusters(sqdist, threshold):
    """Assuming the metric is clustered, return a colored array."""
    # FIXME This is done by components?
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

    def correct(self, f, g):
        """ Check if different methods give the same result. """
        threshold = 1E-10
        print
        for n in range(256, 1024, 128):
            d = np.random.rand(n, n)
            d = d + d.T
            np.fill_diagonal(d, 0)
            d2 = d.copy()
            d3 = d.copy()
            f(d)
            g(d2)
            print "Changed entries: {} out of {}." \
                .format(n*n - np.isclose(d, d3).sum(), n*n)
            error = np.max(np.abs(d-d2))
            print "Absolute difference between methods: ", error
            self.assertLess(error, threshold)
            self.assertTrue(is_metric(d))

    # def test_metricize_pureC(self):
    #     """ Test optimized BLIS metricize against pure C BLIS metricize. """
    #     self.correct(metricize, metricize_pureC)

    def test_metricize_fw(self):
        """ Test Floyd-Warshall metricize against optimized BLIS metricize. """
        self.correct(metricize, metricize_fw)

    def speed(self, f, s=256):
        """ Test speed on larger data sets. """
        print
        for n in range(s, 9*s, s):
            print "Points: ", n
            d = np.random.rand(n, n)
            d = d + d.T
            np.fill_diagonal(d, 0)
            test_speed(f, d, repeat=1)

    def test_speed_metricize(self):
        """ Speed of the optimized BLIS metricize. """
        self.speed(metricize)

    # def test_speed_metricize_pureC(self):
    #     """ Speed of the numpy metricize. """
    #     self.speed(metricize_pureC)

    def test_speed_metricize_fw(self):
        """ Speed of the Floy-Warshall metricize. """
        self.speed(metricize_fw)

if __name__ == "__main__":
    # FIXME add tests for components and is_clustered
    suite = unittest.TestLoader().loadTestsFromTestCase(ToolsTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    exit(0)
    from datetime import datetime
    n = 256
    A = np.random.rand(n, n)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    B = A.copy()
    C = A.copy()
    D = A.copy()
    start = datetime.now()
    metricize_fw(A)
    print datetime.now()-start
    start = datetime.now()
    # add_AB_to_C(B, B, D)
    metricize(B)
    print datetime.now()-start
    # start = datetime.now()
    # metricize5b(A, C)
    # print datetime.now()-start
    print np.max(np.abs(A-B))
