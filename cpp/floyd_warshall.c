// Implementation based on
// http://www.cs.virginia.edu/~pact2006/program/pact2006/pact139_han4.pdf
// mixed with BLIS macro kernel.

// New features:
//  - Only lower triangle of blocks is considered.
//  - Next diagonal block is scheduled as soon as possible, since it is the slowest (real F-W algorithm)
//  - Last diagonal block is generally smaller to handle arbitrary size matrices. 


#include <omp.h>
#include <stdbool.h>
#include <stdio.h> 

// tiles
#define NL1 128
// subtiles
//#define NL2 32
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
#define MAX 1000
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
static double _diag[NL1*NL1] __attribute__ ((aligned (32)));
// next diagonal block
static double _diag2[NL1*NL1] __attribute__ ((aligned (32)));
// horizontal blocks, also used by macro kernel
static double _A[NL1*NL1] __attribute__ ((aligned (32)));
// vertical blocks, also used by macro kernel
static double _B[NL1*NL1] __attribute__ ((aligned (32)));
// temporary matrix for macro kernel
static double _C[MR*NR] __attribute__ ((aligned (32)));
#pragma omp threadprivate(_A, _B, _C)

// BLIS macro kernel
#include "macro.c"

/*
static void show(const double *M, int rows, int cols, int n)
{
    int i, j;
    for (i=0; i<rows;i++)
    {
        for (j=0; j<cols;j++)
            printf("%.2f ", M[i*n+j]);
        printf("\n");
    }
    printf("\n");
}
*/

//
// versions of Floyd-Warshall on up to three matrices
// 

static void diagonal(double *diag, int n)
{
    // Update a diagonal block
    // diag += diag*diag
    // FIXME update lower triangle, then copy results 
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<n;i++)
        {
            double d = diag[i*n+k];
            for (j=0;j<n;j++)
            {
                double s = add(d, diag[k*n+j]);
                diag[i*n+j] = min(diag[i*n+j], s);
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

static void horizontal(const double *diag, double *horz, int rows, int n)
{
    // Update a block from the same row as the current diagonal block
    // horz += diag*horz
    // The horizontal block may have small number of rows, so the diag is also smaller
    int i, j, k;
    for (k=0;k<rows;k++)
        for (i=0;i<rows;i++)
        {
            double d = diag[i*rows+k];
            for (j=0;j<n;j++)
            {
                double s = add(d, horz[k*n+j]);
                horz[i*n+j] = min(horz[i*n+j], s);
            }
        }
}

static void vertical(const double *diag, double *vert, int rows, int n)
{
    // Update a block from the same column as the current diagonal block
    // vert += vert*diag 
    // The vertical block may have small number of rows
    int i, j, k;
    for (k=0;k<n;k++)
        for (i=0;i<rows;i++)
        {
            double v = vert[i*n+k];
            for (j=0;j<n;j++)
            {
                double s = add(v, diag[k*n+j]);
                vert[i*n+j] = min(vert[i*n+j], s);
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
static void pack(double* _M, const double* M, int rows, int cols, int n)
{
    int i, j;
    for (i=0;i<rows;i++)
        for (j=0;j<cols;j++)
            _M[i*cols+j] = M[i*n+j];
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
static void unpack(const double* _M, double* M, int rows, int cols, int n)
{
    int i, j;
    for (i=0;i<rows;i++)
        for (j=0;j<cols;j++)
            M[i*n+j] = _M[i*cols+j];
}

void fw(double *d, const int n)
{
    const int m = (n+NL1-1)/NL1; // number of blocks
    // The last diagonal block might be smaller
    // It has this many rows:
    const int small = n % NL1; 
    // If small > 0 we may be dealing with the last, smaller block
#ifdef NUMCORE
    int numcore = NUMCORE;
#else
    int numcore = omp_get_max_threads();
#endif
    // first diagonal block
    int cols, rows, reduction;
    // It might be small if the whole matrix is small
    rows = n < NL1 ? small : NL1;
    pack(_diag, d, rows, rows, n);
    diagonal(_diag, rows);
    unpack(_diag, d, rows, rows, n);
    if (n < NL1) return;
    int i, j, k;

    for (k=0;k<m;k++)
    {
        // diagonal block already in _diag
#pragma omp parallel default(none) shared(_diag, _diag2, k, d) private(i, j, rows, cols, reduction) num_threads(numcore)
#pragma omp single
 {
        // (k, k+1), (k+1, k) and (k+1, k+1) - new diagonal as soon as possible
        if (k+1<m)
        {
            // FIXME
            // k+1 -> always vertical
            // diagonal based on vertical and transpose of vertical
            // these two could be run by one thread ensuring diagonal is done quickly
            // 
            // We might be working on the last, small row
            rows = ((k+1 < m-1) || (small == 0)) ? NL1 : small;
#pragma omp task depend(inout:d[((k+1)*n+k)*NL1])
  {
            // vertical in _A, diagonal already in _diag
            pack(_A, &d[((k+1)*n+k)*NL1], rows, NL1, n);
            vertical(_diag, _A, rows, NL1);
            unpack(_A, &d[((k+1)*n+k)*NL1], rows, NL1, n);
  }
            
            // now update (k+1, k+1) and run the next diagonal element
#pragma omp task depend(in:d[((k+1)*n+k)*NL1])
  {            
            pack_A(rows, NL1, &d[((k+1)*n+k)*NL1], n, 1, _A);
            pack_B(NL1, rows, &d[((k+1)*n+k)*NL1], 1, n, _B); // transposed A
            dgemm_macro_kernel(rows, rows, NL1, &d[((k+1)*n+k+1)*NL1], 1, n);
            // We can run the next diagonal element
            // Can't be placed in _diag yet
            pack(_diag2, &d[((k+1)*n+k+1)*NL1], rows, rows, n);
            diagonal(_diag2, rows);
            unpack(_diag2, &d[((k+1)*n+k+1)*NL1], rows, rows, n);
  }
        }
        // j < k -> horizontal
        // If k==m-1, we might be updating the last smller row
        rows = ((k < m-1) || (small == 0)) ? NL1 : small;
        for (j=0;j<k;j++)
        {
#pragma omp task depend(inout:d[(k*n+j)*NL1])
  {
            // horizontal in _B, diagonal already in _diag
            pack(_B, &d[(k*n+j)*NL1], rows, NL1, n);
            horizontal(_diag, _B, rows, NL1);
            unpack(_B, &d[(k*n+j)*NL1], rows, NL1, n);
  }
        }
        // j > k -> vertical (k+1 already done)
        for (j=k+2;j<m;j++)
        {
            // The last vertical might be small.
            rows = ((j < m-1) || (small == 0)) ? NL1 : small;
#pragma omp task depend(inout:d[(j*n+k)*NL1])
  {
            // vertical in _A, diagonal already in _diag
            pack(_A, &d[(j*n+k)*NL1], rows, NL1, n);
            vertical(_diag, _A, rows, NL1);
            unpack(_A, &d[(j*n+k)*NL1], rows, NL1, n);
  }
        }
        
        for (i=0;i<m;i++)
        {
            if (i==k) continue;
            // only lower triangle with diagonal (j<=i)
            for (j=0;j<=i;j++)
            {
                if (j==k) continue;
                if ((j==k+1) && (i==k+1)) continue;
                int indexA = j<k ? (k*n+j)*NL1 : (j*n+k)*NL1;
                int indexB = i<k ? (k*n+i)*NL1 : (i*n+k)*NL1;
                // The last row might be small
                rows = ((i < m-1) || (small == 0)) ? NL1 : small;
                // The last column might be small
                cols = ((j < m-1) || (small == 0)) ? NL1 : small;
                // Large tile can be updated from the last small row
                reduction = ((k<m-1) || (small == 0)) ? NL1 : small;
#pragma omp task depend(in:d[indexA], d[indexB]) 
  {
                // other blocks
                // multiplication of transposed matrices to get column order
                // AVX kernel is slightly faster this way
                // also make sure to use only lower triangle (by matrix symmetry)
                // 1, n -> n, 1 in pack_* switches from column to row storage (transposes)
                // transposed horizontal block
                if (j<k)
                    pack_A(NL1, reduction, &d[indexA], 1, n, _A);
                else
                    pack_A(cols, reduction, &d[indexA], n, 1, _A);
                // transposed vertical block
                if (i<k)
                    pack_B(reduction, NL1, &d[indexB], n, 1, _B);
                else
                    pack_B(reduction, rows, &d[indexB], 1, n, _B);
                dgemm_macro_kernel(cols, rows, reduction, &d[(i*n+j)*NL1], 1, n);
  }
            }
        }
 } // single
    for (i=0;i<NL1*NL1;i++)
        _diag[i] = _diag2[i];
    }           
    // copy lower triangle to upper triangle
    for (i=0;i<n;i++)
        for (j=0;j<i;j++)
            d[j*n+i] = d[i*n+j];
}
