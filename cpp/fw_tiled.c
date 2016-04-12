#define NL1 128
#define NL2 32
#define MC NL1
#define NC NL1
#define KC NL1
#define MAX 1e+300

#include "cpp/kernel_BLIS_avx_add.c"

// diagonal block (shared)
static double __C[NL1*NL1] __attribute__ ((aligned (32)));
// horizontal blocks
static double __A[NL1*NL1] __attribute__ ((aligned (32)));
// vertical blocks
static double __B[NL1*NL1] __attribute__ ((aligned (32)));
#pragma omp threadprivate(__A, __B)

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
}
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
}

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

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &__B[j*kc*NR];

        // diagonal blocks could probably run to smaller value
        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &__A[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = __A;
                nextB = &__B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = __B;
                }
            }

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, &__A[i*kc*MR], &__B[j*kc*NR],
                                   false, //full block
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC,
                                   nextA, nextB);
            } 
            /*
            else { // partial block at the edge
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   true, // partial block
                                   _C, 1, MR,
                                   nextA, nextB);
                dgeaxpy(mr, nr, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }*/
        }
    }
}

void show(const double *M, int n)
{
    for (int i=0; i<n;i++)
    {
        for (int j=0; j<n;j++)
            printf("%.2f ", M[i*n+j]);
        printf("\n");
    }
    printf("\n");
}

//
// versions of Floyd-Warshall on up to three matrices
// 

// C <- min(C, C+C)
void fwC(double *C, int n)
{
    for (int k=0;k<n;k++)
        for (int i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (int j=0;j<n;j++)
                if (C[i*n+j] > c+C[k*n+j])
                    C[i*n+j] = c+C[k*n+j];
        }
}

void fwCc(double *C, int n)
{
    // n = 32 ??
    // 8 registers for C[k,:] - read only
    // vbroadcastsd C[i*n+k] - 9th
    // two at a time:
    //   - load C[i, j] (13, 14)
    //   - add (11, 12)
    //   - min (13, 14)
    //   - store 
    for (int k=0;k<n;k++)
        for (int i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (int j=0;j<n;j++)
                if (C[i*n+j] > c+C[k*n+j])
                    C[i*n+j] = c+C[k*n+j];
        }
}

// C <- min(C, A+C)
// FIXME pack A by columns
void fwACC(const double *A, double *C, int n)
{
    for (int k=0;k<n;k++)
        for (int i=0;i<n;i++)
        {
            double a = A[i*n+k];
            for (int j=0;j<n;j++)
                if (C[i*n+j] > a+C[k*n+j])
                    C[i*n+j] = a+C[k*n+j];
        }
}

// C <- min(C, C+B) 
void fwCBC(const double *B, double *C, int n)
{
    for (int k=0;k<n;k++)
        for (int i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (int j=0;j<n;j++)
                if (C[i*n+j] > c+B[k*n+j])
                    C[i*n+j] = c+B[k*n+j];
        }
}

/*
// C <- min(C, A+B) - tropical matrix multiplication
// B packed by columns
void fwABC(const double *A, const double *B, double *C, int n, int full_n)
{
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
        {
            double c = C[i*full_n+j];
            for (int k=0;k<n;k++)
                if (c > A[i*n+k]+B[j*n+k])
                    c = A[i*n+k]+B[j*n+k];
            C[i*full_n+j] = c;
        }
}
*/

// move a block into contiguous memory
void pack(double* _M, const double* M, int n)
{
    for (int i=0;i<NL1;i++)
        for (int j=0;j<NL1;j++)
            _M[i*NL1+j] = M[i*n+j];
}

/*
// move a block into contiguous memory column first
void pack_col(double* _M,const  double* M, int n)
{
    for (int j=0;j<NL1;j++)
        for (int i=0;i<NL1;i++)
            _M[j*NL1+i] = M[i*n+j];
}
*/

// move a block back to its place 
void unpack(const double* _M, double* M, int n)
{
    for (int i=0;i<NL1;i++)
        for (int j=0;j<NL1;j++)
            M[i*n+j] = _M[i*NL1+j];
}

void fw(double *d, const int n)
{
    const int m = n/NL1; // number of blocks
    // first diagonal block
    pack(__C, d, n);
    fwC(__C, NL1);
    unpack(__C, d, n);
    for (int k=0;k<m;k++)
    {
        // diagonal block already in __C
#pragma omp parallel default(none) shared(__C, k, d)
{
#pragma omp single
 {
        for (int j=0;j<m;j++)
        {
            if (j==k) continue;
            // horizontal in __B, diagonal already in __C
#pragma omp task
  {
            pack(__B, &d[(k*n+j)*NL1], n);
            fwACC(__C, __B, NL1);
            unpack(__B, &d[(k*n+j)*NL1], n);
  }
            // vertical in __A, diagonal already in __C
#pragma omp task
  {
            pack(__A, &d[(j*n+k)*NL1], n);
            fwCBC(__C, __A, NL1);
            unpack(__A, &d[(j*n+k)*NL1], n);
  }
        }
#pragma omp taskwait
        // FIXME get rid of taskwait in favor of depend
        for (int i=0;i<m;i++)
        {
            if (i==k) continue;
            for (int j=0;j<m;j++)
            {
                if (j==k) continue;
                // other blocks
#pragma omp task
  {
                //pack_col(__B, &d[(k*n+j)*NL1], n);
                //pack(__A, &d[(i*n+k)*NL1], n);
                // no packing on the result
                //fwABC(__A, __B, &d[(i*n+j)*NL1], NL1, n);
                pack_B(NL1, NL1, &d[(k*n+j)*NL1], n, 1, __B);
                pack_A(NL1, NL1, &d[(i*n+k)*NL1], n, 1, __A);
                dgemm_macro_kernel(NL1, NL1, NL1, &d[(i*n+j)*NL1], n, 1);
                if ((i==k+1) && (j==k+1))
                {
                    // We can run the next diagonal element
                    pack(__C, &d[(i*n+i)*NL1], n);
                    fwC(__C, NL1);
                    unpack(__C, &d[(i*n+i)*NL1], n);
                }
  }
            }
        }
 } // single
} // parallel
    }           
}
