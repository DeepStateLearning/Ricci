// A few definitions relevant to tropical matrix multiplication

#define MC  96
#define KC  256
#define NC  4096

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




//
//  Local buffers for storing panels from A, B and C
//  change depending on kernel
//
#if defined __FAST_MATH__
#include "kernel_ulmBLAS_pureC.c"
// there is a chance compiler will optimize
static double _A[MC*KC] __attribute__ ((aligned (32)));
static double _B[KC*NC] __attribute__ ((aligned (32)));
static double _C[MR*NR] __attribute__ ((aligned (32)));
#elif defined __AVX__
#include "kernel_BLIS_avx.c"
static double _A[MC*KC] __attribute__ ((aligned (32)));
static double _B[KC*NC] __attribute__ ((aligned (32)));
static double _C[MR*NR] __attribute__ ((aligned (32)));
#elif defined __SSE3__
#include "kernel_ulmBLAS_sse.c"
static double _A[MC*KC] __attribute__ ((aligned (16)));
static double _B[KC*NC] __attribute__ ((aligned (16)));
static double _C[MR*NR] __attribute__ ((aligned (16)));
#else 
#include "kernel_ulmBLAS_pureC.c"
static double _A[MC*KC];
static double _B[KC*NC];
static double _C[MR*NR];
#endif

