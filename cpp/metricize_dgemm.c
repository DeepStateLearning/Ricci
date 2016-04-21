#include "genmul.h"

#if defined __FAST_MATH__
void metricize_pure(double* d, double* d2, int n, int limit)
#else
void metricize(double* d, double* d2, int n, int limit)
#endif
{
    int i, count = 1;
    double error = 1.0;
    while (error > 10e-12)
    {
        error = 0;
        // shortest paths to d2
#if defined __FAST_MATH__
        dgemm_pure(n, d, d2);
#else
        dgemm_nn(n, d, d2);
#endif

#pragma omp parallel for private(i) shared(d, d2) reduction(max:error)
        for (i=0;i<n*n; i++)
        {
            if (error<d[i]-d2[i])
                error=d[i]-d2[i];
            d[i] = d2[i];
        }
    
        if ((limit > 0) && (++count > limit)) break;
    }
}
