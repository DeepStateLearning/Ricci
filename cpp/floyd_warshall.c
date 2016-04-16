// Implementation based on
// http://www.cs.virginia.edu/~pact2006/program/pact2006/pact139_han4.pdf
// mixed with BLIS macro kernel.

#include <omp.h>
#include <stdbool.h>
#include <stdio.h> 

// tiles
#define NL1 128
// subtiles
#define NL2 32
// BLIS macro kernel uses large tiles
#define MC NL1
#define NC NL1
#define KC NL1

// generalized inner product
// new reduction operation
#define min(x,y)  (((x)<(y)) ? (x) : (y))
#define asmmin minpd
// zero element of the semiring
// e.g. infinity for min and plus/times
//      0 for plus and times
#define MAX 1e+300
// unit element for reduction
#define UNIT 1.0
// new pointwise operation
#define add(x,y)  ((x)*(y))
#define asmadd mulpd
// redefine these to get other generalized inner products
// for assembly, provide SSE instructions
// avx versions (prefixed with v) are automatically created in AVX kernel

// right now only AVX kernel
#include "kernel_BLIS_avx.c"

// diagonal block (shared)
static double _C[NL1*NL1] __attribute__ ((aligned (32)));
// next diagonal block
static double _CC[NL1*NL1] __attribute__ ((aligned (32)));
// horizontal blocks
static double _A[NL1*NL1] __attribute__ ((aligned (32)));
// vertical blocks
static double _B[NL1*NL1] __attribute__ ((aligned (32)));
#pragma omp threadprivate(_A, _B)

// BLIS macro kernel
#include "macro.c"

/*
static void show(const double *M, int n)
{
    int i, j;
    for (i=0; i<n;i++)
    {
        for (j=0; j<n;j++)
            printf("%.2f ", M[i*n+j]);
        printf("\n");
    }
    printf("\n");
}
*/

//
// versions of Floyd-Warshall on up to three matrices
// 

// C <- min(C, C+C)
static void fwC(double *C, int n)
{
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (j=0;j<n;j++)
            {
                double s = add(c, C[k*n+j]);
                C[i*n+j] = min(C[i*n+j], s);
            }
        }
}

/*
static void fwCc(double *C, int n)
{
    // n = 32 ??
    // 8 registers for C[k,:] - read only
    // vbroadcastsd C[i*n+k] - 9th
    // two at a time:
    //   - load C[i, j] (13, 14)
    //   - add (11, 12)
    //   - min (13, 14)
    //   - store 
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (j=0;j<n;j++)
                C[i*n+j] = min(C[i*n+j], add(c, C[k*n+j]));
        }
}
*/

// C <- min(C, A+C)
// FIXME pack A by columns
static void fwACC(const double *A, double *C, int n)
{
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<n;i++)
        {
            double a = A[i*n+k];
            for (j=0;j<n;j++)
            {
                double s = add(a, C[k*n+j]);
                C[i*n+j] = min(C[i*n+j], s);
            }
        }
}

// C <- min(C, C+B) 
static void fwCBC(const double *B, double *C, int n)
{
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<n;i++)
        {
            double c = C[i*n+k];
            for (j=0;j<n;j++)
            {
                double s = add(c, B[k*n+j]);
                C[i*n+j] = min(C[i*n+j], s);
            }
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
static void pack(double* _M, const double* M, int n)
{
    int i, j;
    for (i=0;i<NL1;i++)
        for (j=0;j<NL1;j++)
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
static void unpack(const double* _M, double* M, int n)
{
    int i, j;
    for (i=0;i<NL1;i++)
        for (j=0;j<NL1;j++)
            M[i*n+j] = _M[i*NL1+j];
}

void fw(double *d, const int n)
{
    const int m = n/NL1; // number of blocks
#ifdef NUMCORE
    int numcore = NUMCORE;
#else
    int numcore = omp_get_max_threads();
#endif
    // first diagonal block
    pack(_C, d, n);
    fwC(_C, NL1);
    unpack(_C, d, n);
    int i, j, k;
    for (k=0;k<m;k++)
    {
        // diagonal block already in _C
#pragma omp parallel default(none) shared(_C, _CC, k, d) private(i, j) num_threads(numcore)
{
#pragma omp single
 {
        // (k, k+1), (k+1, k) and (k+1, k+1) - new diagonal as soon as possible
        // FIXME how to force it to run as sson as possible after its dependencies?
        if (k+1<m)
        {
#pragma omp task depend(out:d[(k*n+k+1)*NL1])
  {
            // horizontal in _B, diagonal already in _C
            pack(_B, &d[(k*n+k+1)*NL1], n);
            fwACC(_C, _B, NL1);
            unpack(_B, &d[(k*n+k+1)*NL1], n);
  }
#pragma omp task depend(out:d[((k+1)*n+k)*NL1])
  {
            // vertical in _A, diagonal already in _C
            pack(_A, &d[((k+1)*n+k)*NL1], n);
            fwCBC(_C, _A, NL1);
            unpack(_A, &d[((k+1)*n+k)*NL1], n);
  }
#pragma omp task depend(in:d[(k*n+k+1)*NL1], d[((k+1)*n+k)*NL1])
  {
            pack_B(NL1, NL1, &d[(k*n+k+1)*NL1], n, 1, _B);
            pack_A(NL1, NL1, &d[((k+1)*n+k)*NL1], n, 1, _A);
            dgemm_macro_kernel(NL1, NL1, NL1, &d[((k+1)*n+k+1)*NL1], n, 1);
            // We can run the next diagonal element
            // Can't be placed in _C yet
            pack(_CC, &d[((k+1)*n+k+1)*NL1], n);
            fwC(_CC, NL1);
            unpack(_CC, &d[((k+1)*n+k+1)*NL1], n);
  }
        }
        for (j=0;j<m;j++)
        {
            if ((j==k) || (j==k+1)) continue;
#pragma omp task depend(out:d[(k*n+j)*NL1])
  {
            // horizontal in _B, diagonal already in _C
            pack(_B, &d[(k*n+j)*NL1], n);
            fwACC(_C, _B, NL1);
            unpack(_B, &d[(k*n+j)*NL1], n);
  }
#pragma omp task depend(out:d[(j*n+k)*NL1])
  {
            // vertical in _A, diagonal already in _C
            pack(_A, &d[(j*n+k)*NL1], n);
            fwCBC(_C, _A, NL1);
            unpack(_A, &d[(j*n+k)*NL1], n);
  }
        }
        for (i=0;i<m;i++)
        {
            if (i==k) continue;
            for (j=0;j<m;j++)
            {
                if (j==k) continue;
                if ((j==k+1) && (i==k+1)) continue;
#pragma omp task depend(in:d[(k*n+j)*NL1], d[(i*n+k)*NL1])
  {
                // other blocks
                pack_B(NL1, NL1, &d[(k*n+j)*NL1], n, 1, _B);
                pack_A(NL1, NL1, &d[(i*n+k)*NL1], n, 1, _A);
                dgemm_macro_kernel(NL1, NL1, NL1, &d[(i*n+j)*NL1], n, 1);
  }
            }
        }
 } // single
} // parallel
    for (i=0;i<NL1*NL1;i++)
        _C[i] = _CC[i];
    }           
}
