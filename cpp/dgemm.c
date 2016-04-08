// dgemm turned into metricize as weave extension
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
#define MC  96
#define KC  256
#define NC  4096

// generalized inner product
// new reduction operation
#define min(x,y)  (((x)<(y)) ? (x) : (y))
// zero element for the reduction
#define MAX 1e+300
// new pointwise operation
#define add(x,y)  ((x)*(y))
// redefine these to get other generalized inner products in pure C
// inline assembly requires changes by hand, 
// e.g. addpd->minpd and mulpd->addpd in our case
// max is automatically embedded, but also not necessary if MAX=0

//
//  Local buffers for storing panels from A, B and C
//  change depending on kernel
//
#if defined __FAST_MATH__
#include "cpp/kernel_ulmBLAS_pureC.c"
#elif defined __AVX__
#include "cpp/kernel_BLIS_avx.c"
#elif defined __SSE3__
#include "cpp/kernel_ulmBLAS_sse.c"
#else 
#include "cpp/kernel_ulmBLAS_pureC.c"
#endif

#pragma omp threadprivate(_C)
#pragma omp threadprivate(_A)

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const double *A, int incRowA, int incColA,
          double *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const double *A, int incRowA, int incColA,
       double *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = MAX;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const double *B, int incRowB, int incColB,
          double *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const double *B, int incRowB, int incColB,
       double *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = MAX;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

//
//  Compute Y += alpha*X
//
static void
dgeaxpy(int           m,
        int           n,
        const double  *X,
        int           incRowX,
        int           incColX,
        double        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;
    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
            // omp atomic here? but how with min() and Y on both sides
            while (Y[i*incRowY+j*incColY] > X[i*incRowX+j*incColX])
                Y[i*incRowY+j*incColY] = X[i*incRowX+j*incColX];
        }
    }
}

//
//  Compute X *= alpha
//

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   double  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    const double *nextA;
    const double *nextB;

//#pragma omp parallel for default(shared) private(j, i, mr, nr, nextA, nextB) num_threads(8)
    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*kc*NR];

        // diagonal blocks could probably run to smaller value
        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &_A[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = _A;
                nextB = &_B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   false, //full block
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC,
                                   nextA, nextB);
            } else { // partial block at the edge
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   true, // partial block
                                   _C, 1, MR,
                                   nextA, nextB);
                dgeaxpy(mr, nr, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
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

