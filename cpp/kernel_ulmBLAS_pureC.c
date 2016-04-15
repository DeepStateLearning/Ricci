// Modified fully optimized ulmBLAS micro kernel
#define MR  4
#define NR  4

//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(long kc,
                   const double *A, const double *B,
                   bool partial,
                   double *C, long incRowC, long incColC,
                   const double *nextA, const double *nextB)
{

    int i, j, l;
    double AB[MR*NR];

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
    if (partial) {
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
            if (C[i*incRowC+j*incColC] > AB[i+j*MR])
                C[i*incRowC+j*incColC] = AB[i+j*MR];
        }
    }
}
