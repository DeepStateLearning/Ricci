// dgemm turned into generalized matrix multiplication
// Computes C += A*B
// but with pointwise * replaced with +
// and + replaced with min(x,y)
//
// Other changes:
// - added OpenMp in the first loop outside of the macro kernel
//      so that _A is in private L2 cache for each core.
// - incorporates three kernels
//      * pure C from ulmBLAS (used with ffast-math and when no SSE 
//      * fully optimized asm SSE from ulmBLAS (when AVX not available)
//      * asm avx kernel from BLIS (when avx is available)
//   Unfortunately avx2 will not be of much help, 
//   since fused madd cannot be used with min-add reduction
// 
// Based on dgemm from:
//      https://github.com/michael-lehn/ulmBLAS
// The above implements the BLIS framework:
//      https://github.com/flame/blis
//
// FIXME switch to syrk for better performance

#include <omp.h>
#include <stdbool.h>

#include "genmul_defs.h"

#pragma omp threadprivate(_C)
#pragma omp threadprivate(_A)

#include "macro.c"

//
//  Compute C <- C + A*B
//
void
dgemm_nn(int            n,
         const double   *A,
         double         *C)
{
    int m = n, k = n;
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int incRowA = 1, incRowB = 1, incRowC = 1;
    int incColA = n, incColB = n, incColC = n;
    const double *B = A;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    // gemm should not be using hyperthreading
#ifdef NUMCORE
    int numcore = NUMCORE;
#else
    int numcore = omp_get_max_threads();
#endif

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);
#pragma omp parallel for default(shared) private(i, mc) num_threads(numcore)
            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc, 
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }
}

