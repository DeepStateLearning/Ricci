// dgemm turned into metricize as weave extension
// Computes C += A*B
// but with pointwise * replaced with +
// and + replaced with min(x,y)
//
// Other changes:
// - added OpenMp in macro kernel
// - micro kernel not fully optimized (pure c, no SSE, no asm)
// 
// Based on pure-c dgemm from:
//      https://github.com/michael-lehn/ulmBLAS
// The above implements the BLIS framework:
//      https://github.com/flame/blis
#define MC  384
#define KC  384
#define NC  4096

#define MR  4
#define NR  4

// generalized inner product
// new reduction operation
#define min(x,y)  (((x)<(y)) ? (x) : (y))
// zero element for the reduction
#define MAX 1.7976931348623158e+308
// new pointwise operation
#define add(x,y)  ((x)+(y))
// redefine these to get other generalized inner products

//
//  Local buffers for storing panels from A, B and C
//
static double _A[MC*KC];
static double _B[KC*NC];
static double _C[MR*NR];

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
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(int kc,
                   const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC)
{
    double AB[MR*NR];

    int i, j, l;

//
//  Compute AB = A*B
//
    for (j=0; j<NR*MR; ++j) {
            AB[j] = MAX;
    }
    for (l=0; l<kc; ++l) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                AB[i+j*MR] = min(add(A[i], B[j]), AB[i+j*MR]);
            }
        }
        A += MR;
        B += NR;
    }

//
//  Update C <- beta*C
//
    if (beta==0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = MAX;
            }
        }
    } 

//
//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
//                                  the above layer dgemm_nn)
//
    for (j=0; j<NR; ++j) {
        for (i=0; i<MR; ++i) {
            C[i*incRowC+j*incColC] = min(AB[i+j*MR], C[i*incRowC+j*incColC]);
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
            Y[i*incRowY+j*incColY] = min(X[i*incRowX+j*incColX], Y[i*incRowY+j*incColY]);
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

#pragma omp parallel for default(shared) private(i, mr, nr)
    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   1.0,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0.0, // zero element of the reduction operation
                                   _C, 1, MR);
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
dgemm_nn(int            m,
         int            n,
         int            k,
         const double   *A,
         int            incRowA,
         int            incColA,
         const double   *B,
         int            incRowB,
         int            incColB,
         double         *C,
         int            incRowC,
         int            incColC)
{
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

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

