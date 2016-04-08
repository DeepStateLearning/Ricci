//
// Modified fully optimized ulmBLAS micro kernel
//
// - addpd replaced with minpd as reduction operation
// - xorps replaced with load MAX value to initialize registers
#define MR  4
#define NR  4

static double _A[MC*KC] __attribute__ ((aligned (16)));
static double _B[KC*NC] __attribute__ ((aligned (16)));
static double _C[MR*NR] __attribute__ ((aligned (16)));

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
    long kb = kc / 4;
    long kl = kc % 4;
    double beta = partial ? MAX : 0.0;
    double alpha = MAX; // used to transfer MAX to registers
//
//  Compute AB = A*B
//
__asm__ volatile
    (
    "movq      %0,      %%rsi    \n\t"  // kb (32 bit) stored in %rsi
    "movq      %1,      %%rdi    \n\t"  // kl (32 bit) stored in %rdi
    "movq      %2,      %%rax    \n\t"  // Address of A stored in %rax
    "movq      %3,      %%rbx    \n\t"  // Address of B stored in %rbx
    "movq      %9,      %%r9     \n\t"  // Address of nextA stored in %r9
    "movq      %10,     %%r10    \n\t"  // Address of nextB stored in %r10
    "                            \n\t"
    "addq      $128,    %%rax    \n\t"
    "addq      $128,    %%rbx    \n\t"
    "                            \n\t"
    "movapd -128(%%rax),%%xmm0   \n\t"  // tmp0 = _mm_load_pd(A)
    "movapd -112(%%rax),%%xmm1   \n\t"  // tmp1 = _mm_load_pd(A+2)
    "movapd -128(%%rbx),%%xmm2   \n\t"  // tmp2 = _mm_load_pd(B)
    "                            \n\t"
    "movsd  %4,         %%xmm8   \n\t"  // load alpha
    "unpcklpd  %%xmm8,  %%xmm8   \n\t"  // ab_00_11 = MAX
    "movapd    %%xmm8,  %%xmm9   \n\t"  // ab_20_31 = MAX
    "movapd    %%xmm8,  %%xmm10  \n\t"  // ab_01_10 = MAX
    "movapd    %%xmm9,  %%xmm11  \n\t"  // ab_21_30 = MAX
    "movapd    %%xmm8,  %%xmm12  \n\t"  // ab_02_13 = MAX
    "movapd    %%xmm9,  %%xmm13  \n\t"  // ab_22_33 = MAX
    "movapd    %%xmm10, %%xmm14  \n\t"  // ab_03_12 = MAX
    "movapd    %%xmm11, %%xmm15  \n\t"  // ab_23_32 = MAX
    "                            \n\t"
    "movapd    %%xmm8,  %%xmm3   \n\t"  // tmp3 = MAX
    "movapd    %%xmm9,  %%xmm4   \n\t"  // tmp4 = MAX
    "movapd    %%xmm10, %%xmm5   \n\t"  // tmp5 = MAX
    "movapd    %%xmm11, %%xmm6   \n\t"  // tmp6 = MAX
    "movapd    %%xmm12, %%xmm7   \n\t"  // tmp7 = MAX
    "testq     %%rdi,   %%rdi    \n\t"  // if kl==0 writeback to AB
    "                            \n\t"
    "                            \n\t"
    "testq     %%rsi,   %%rsi    \n\t"  // if kb==0 handle remaining kl
    "je        .DCONSIDERLEFT%=  \n\t"  // update iterations
    "                            \n\t"
    ".DLOOP%=:                   \n\t"  // for l = kb,..,1 do
    "                            \n\t"
    "prefetcht0 (4*35+1)*8(%%rax)\n\t"
    "                            \n\t"
    "                            \n\t"  // 1. update
    "minpd     %%xmm3,  %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movaps -112(%%rbx),%%xmm3   \n\t"  // tmp3     = _mm_load_pd(B+2)
    "minpd     %%xmm6,  %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movaps    %%xmm2,  %%xmm6   \n\t"  // tmp6     = tmp2
    "pshufd $78,%%xmm2, %%xmm4   \n\t"  // tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm2   \n\t"  // tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm5,  %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd     %%xmm7,  %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movaps    %%xmm4,  %%xmm7   \n\t"  // tmp7     = tmp4
    "mulpd     %%xmm0,  %%xmm4   \n\t"  // tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm2,  %%xmm8   \n\t"  // ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movaps -96(%%rbx), %%xmm2   \n\t"  // tmp2     = _mm_load_pd(B+4)
    "minpd     %%xmm6,  %%xmm9   \n\t"  // ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movaps    %%xmm3,  %%xmm6   \n\t"  // tmp6     = tmp3
    "pshufd $78,%%xmm3, %%xmm5   \n\t"  // tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm3   \n\t"  // tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm4,  %%xmm10  \n\t"  // ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "minpd     %%xmm7,  %%xmm11  \n\t"  // ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movaps    %%xmm5,  %%xmm7   \n\t"  // tmp7     = tmp5
    "mulpd     %%xmm0,  %%xmm5   \n\t"  // tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movaps -96(%%rax), %%xmm0   \n\t"  // tmp0     = _mm_load_pd(A+4)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movaps -80(%%rax), %%xmm1   \n\t"  // tmp1     = _mm_load_pd(A+6)
    "                            \n\t"
    "                            \n\t"
    "                            \n\t"
    "                            \n\t"  // 2. update
    "minpd     %%xmm3,  %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movaps -80(%%rbx), %%xmm3   \n\t"  // tmp3     = _mm_load_pd(B+6)
    "minpd     %%xmm6,  %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movaps    %%xmm2,  %%xmm6   \n\t"  // tmp6     = tmp2
    "pshufd $78,%%xmm2, %%xmm4   \n\t"  // tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm2   \n\t"  // tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm5,  %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd     %%xmm7,  %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movaps    %%xmm4,  %%xmm7   \n\t"  // tmp7     = tmp4
    "mulpd     %%xmm0,  %%xmm4   \n\t"  // tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm2,  %%xmm8   \n\t"  // ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movaps -64(%%rbx), %%xmm2   \n\t"  // tmp2     = _mm_load_pd(B+8)
    "minpd     %%xmm6,  %%xmm9   \n\t"  // ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movaps    %%xmm3,  %%xmm6   \n\t"  // tmp6     = tmp3
    "pshufd $78,%%xmm3, %%xmm5   \n\t"  // tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm3   \n\t"  // tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm4,  %%xmm10  \n\t"  // ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "minpd     %%xmm7,  %%xmm11  \n\t"  // ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movaps    %%xmm5,  %%xmm7   \n\t"  // tmp7     = tmp5
    "mulpd     %%xmm0,  %%xmm5   \n\t"  // tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movaps -64(%%rax), %%xmm0   \n\t"  // tmp0     = _mm_load_pd(A+8)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movaps -48(%%rax), %%xmm1   \n\t"  // tmp1     = _mm_load_pd(A+10)
    "                            \n\t"
    "                            \n\t"
    "prefetcht0 (4*37+1)*8(%%rax)\n\t"
    "                            \n\t"
    "                            \n\t"  // 3. update
    "minpd     %%xmm3,  %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movaps -48(%%rbx), %%xmm3   \n\t"  // tmp3     = _mm_load_pd(B+10)
    "minpd     %%xmm6,  %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movaps    %%xmm2,  %%xmm6   \n\t"  // tmp6     = tmp2
    "pshufd $78,%%xmm2, %%xmm4   \n\t"  // tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm2   \n\t"  // tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm5,  %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd     %%xmm7,  %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movaps    %%xmm4,  %%xmm7   \n\t"  // tmp7     = tmp4
    "mulpd     %%xmm0,  %%xmm4   \n\t"  // tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm2,  %%xmm8   \n\t"  // ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movaps -32(%%rbx), %%xmm2   \n\t"  // tmp2     = _mm_load_pd(B+12)
    "minpd     %%xmm6,  %%xmm9   \n\t"  // ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movaps    %%xmm3,  %%xmm6   \n\t"  // tmp6     = tmp3
    "pshufd $78,%%xmm3, %%xmm5   \n\t"  // tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm3   \n\t"  // tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm4,  %%xmm10  \n\t"  // ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "minpd     %%xmm7,  %%xmm11  \n\t"  // ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movaps    %%xmm5,  %%xmm7   \n\t"  // tmp7     = tmp5
    "mulpd     %%xmm0,  %%xmm5   \n\t"  // tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movaps -32(%%rax), %%xmm0   \n\t"  // tmp0     = _mm_load_pd(A+12)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movaps -16(%%rax), %%xmm1   \n\t"  // tmp1     = _mm_load_pd(A+14)
    "                            \n\t"
    "                            \n\t"
    "                            \n\t"  // 4. update
    "minpd     %%xmm3,  %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movaps -16(%%rbx), %%xmm3   \n\t"  // tmp3     = _mm_load_pd(B+14)
    "minpd     %%xmm6,  %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movaps    %%xmm2,  %%xmm6   \n\t"  // tmp6     = tmp2
    "pshufd $78,%%xmm2, %%xmm4   \n\t"  // tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm2   \n\t"  // tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"
    "subq     $-32*4,   %%rax    \n\t"  // A += 16;
    "                            \n\t"
    "minpd     %%xmm5,  %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd     %%xmm7,  %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movaps    %%xmm4,  %%xmm7   \n\t"  // tmp7     = tmp4
    "mulpd     %%xmm0,  %%xmm4   \n\t"  // tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"
    "subq     $-128,    %%r10    \n\t"  // nextB += 16
    "                            \n\t"
    "minpd     %%xmm2,  %%xmm8   \n\t"  // ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movaps    (%%rbx),%%xmm2   \n\t"  // tmp2     = _mm_load_pd(B+16)
    "minpd     %%xmm6,  %%xmm9   \n\t"  // ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movaps    %%xmm3,  %%xmm6   \n\t"  // tmp6     = tmp3
    "pshufd $78,%%xmm3, %%xmm5   \n\t"  // tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm3   \n\t"  // tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"
    "subq     $-32*4,   %%rbx    \n\t"  // B += 16;
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm4,  %%xmm10  \n\t"  // ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "minpd     %%xmm7,  %%xmm11  \n\t"  // ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movaps    %%xmm5,  %%xmm7   \n\t"  // tmp7     = tmp5
    "mulpd     %%xmm0,  %%xmm5   \n\t"  // tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movaps -128(%%rax),%%xmm0   \n\t"  // tmp0     = _mm_load_pd(A+16)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movaps -112(%%rax), %%xmm1   \n\t"  // tmp1     = _mm_load_pd(A+18)
    "                            \n\t"
    "prefetcht2        0(%%r10)  \n\t"  // prefetch nextB[0]
    "prefetcht2       64(%%r10)  \n\t"  // prefetch nextB[8]
    "                            \n\t"
    "decq      %%rsi             \n\t"  // --l
    "jne       .DLOOP%=          \n\t"  // if l>= 1 go back
    "                            \n\t"
    "                            \n\t"
    ".DCONSIDERLEFT%=:           \n\t"
    "testq     %%rdi,   %%rdi    \n\t"  // if kl==0 writeback to AB
    "je        .DPOSTACCUMULATE%=\n\t"
    "                            \n\t"
    ".DLOOPLEFT%=:               \n\t"  // for l = kl,..,1 do
    "                            \n\t"
    "minpd     %%xmm3,  %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "movapd -112(%%rbx),%%xmm3   \n\t"  // tmp3     = _mm_load_pd(B+2)
    "minpd     %%xmm6,  %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "movapd    %%xmm2,  %%xmm6   \n\t"  // tmp6     = tmp2
    "pshufd $78,%%xmm2, %%xmm4   \n\t"  // tmp4     = _mm_shuffle_pd(tmp2, tmp2,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm2   \n\t"  // tmp2     = _mm_mul_pd(tmp2, tmp0);
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1);
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm5,  %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd     %%xmm7,  %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "movapd    %%xmm4,  %%xmm7   \n\t"  // tmp7     = tmp4
    "mulpd     %%xmm0,  %%xmm4   \n\t"  // tmp4     = _mm_mul_pd(tmp4, tmp0)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm2,  %%xmm8   \n\t"  // ab_00_11 = _mm_add_pd(ab_00_11, tmp2)
    "movapd -96(%%rbx), %%xmm2   \n\t"  // tmp2     = _mm_load_pd(B+4)
    "minpd     %%xmm6,  %%xmm9   \n\t"  // ab_20_31 = _mm_add_pd(ab_20_31, tmp6)
    "movapd    %%xmm3,  %%xmm6   \n\t"  // tmp6     = tmp3
    "pshufd $78,%%xmm3, %%xmm5   \n\t"  // tmp5     = _mm_shuffle_pd(tmp3, tmp3,
    "                            \n\t"  //                   _MM_SHUFFLE2(0, 1))
    "mulpd     %%xmm0,  %%xmm3   \n\t"  // tmp3     = _mm_mul_pd(tmp3, tmp0)
    "mulpd     %%xmm1,  %%xmm6   \n\t"  // tmp6     = _mm_mul_pd(tmp6, tmp1)
    "                            \n\t"
    "                            \n\t"
    "minpd     %%xmm4,  %%xmm10  \n\t"  // ab_01_10 = _mm_add_pd(ab_01_10, tmp4)
    "minpd     %%xmm7,  %%xmm11  \n\t"  // ab_21_30 = _mm_add_pd(ab_21_30, tmp7)
    "movapd    %%xmm5,  %%xmm7   \n\t"  // tmp7     = tmp5
    "mulpd     %%xmm0,  %%xmm5   \n\t"  // tmp5     = _mm_mul_pd(tmp5, tmp0)
    "movapd -96(%%rax), %%xmm0   \n\t"  // tmp0     = _mm_load_pd(A+4)
    "mulpd     %%xmm1,  %%xmm7   \n\t"  // tmp7     = _mm_mul_pd(tmp7, tmp1)
    "movapd -80(%%rax), %%xmm1   \n\t"  // tmp1     = _mm_load_pd(A+6)
    "                            \n\t"
    "                            \n\t"
    "addq      $32,     %%rax    \n\t"  // A += 4;
    "addq      $32,     %%rbx    \n\t"  // B += 4;
    "                            \n\t"
    "decq      %%rdi             \n\t"  // --l
    "jne       .DLOOPLEFT%=      \n\t"  // if l>= 1 go back
    "                            \n\t"
    ".DPOSTACCUMULATE%=:         \n\t"  // Update remaining ab_*_* registers
    "                            \n\t"
    "minpd    %%xmm3,   %%xmm12  \n\t"  // ab_02_13 = _mm_add_pd(ab_02_13, tmp3)
    "minpd    %%xmm6,   %%xmm13  \n\t"  // ab_22_33 = _mm_add_pd(ab_22_33, tmp6)
    "                            \n\t"  //
    "minpd    %%xmm5,   %%xmm14  \n\t"  // ab_03_12 = _mm_add_pd(ab_03_12, tmp5)
    "minpd    %%xmm7,   %%xmm15  \n\t"  // ab_23_32 = _mm_add_pd(ab_23_32, tmp7)
    "                            \n\t"
//
//  Update C <- beta*C + alpha*AB
//
//
    "movsd  %5,                  %%xmm1 \n\t"  // load beta 
    "movq   %6,                  %%rcx  \n\t"  // Address of C stored in %rcx

    "movq   %7,                  %%r8   \n\t"  // load incRowC
    "leaq   (,%%r8,8),           %%r8   \n\t"  //      incRowC *= sizeof(double)
    "movq   %8,                  %%r9   \n\t"  // load incColC
    "leaq   (,%%r9,8),           %%r9   \n\t"  //      incRowC *= sizeof(double)
    "                                   \n\t"
    "leaq (%%rcx,%%r9),          %%r10  \n\t"  // Store addr of C01 in %r10
    "leaq (%%rcx,%%r8,2),        %%rdx  \n\t"  // Store addr of C20 in %rdx
    "leaq (%%rdx,%%r9),          %%r11  \n\t"  // Store addr of C21 in %r11
    "                                   \n\t"
    "unpcklpd %%xmm1,            %%xmm1 \n\t"  // duplicate beta
    "                                   \n\t"
    "                                   \n\t"
    "movlpd (%%rcx),             %%xmm3 \n\t"  // load (C00,
    "movhpd (%%r10,%%r8),        %%xmm3 \n\t"  //       C11)
    //"mulpd  %%xmm0,              %%xmm8 \n\t"  // scale ab_00_11 by alpha
    "mulpd  %%xmm1,              %%xmm3 \n\t"  // scale (C00, C11) by beta
    "minpd  %%xmm8,              %%xmm3 \n\t"  // add results

    "movlpd %%xmm3,        (%%rcx)       \n\t"  // write back (C00,
    "movhpd %%xmm3,        (%%r10,%%r8)  \n\t"  //             C11)
    "                                   \n\t"
    "movlpd (%%rdx),             %%xmm4 \n\t"  // load (C20,
    "movhpd (%%r11,%%r8),        %%xmm4 \n\t"  //       C31)
    //"mulpd  %%xmm0,              %%xmm9 \n\t"  // scale ab_20_31 by alpha
    "mulpd  %%xmm1,              %%xmm4 \n\t"  // scale (C20, C31) by beta
    "minpd  %%xmm9,              %%xmm4 \n\t"  // add results
    "movlpd %%xmm4,        (%%rdx)       \n\t"  // write back (C20,
    "movhpd %%xmm4,        (%%r11,%%r8)  \n\t"  //             C31)
    "                                   \n\t"
    "                                   \n\t"
    "movlpd (%%r10),             %%xmm3 \n\t"  // load (C01,
    "movhpd (%%rcx,%%r8),        %%xmm3 \n\t"  //       C10)
    //"mulpd  %%xmm0,              %%xmm10\n\t"  // scale ab_01_10 by alpha
    "mulpd  %%xmm1,              %%xmm3 \n\t"  // scale (C01, C10) by beta
    "minpd  %%xmm10,             %%xmm3 \n\t"  // add results
    "movlpd %%xmm3,        (%%r10)      \n\t"  // write back (C01,
    "movhpd %%xmm3,        (%%rcx,%%r8) \n\t"  //             C10)
    "                                   \n\t"
    "movlpd (%%r11),             %%xmm4 \n\t"  // load (C21,
    "movhpd (%%rdx,%%r8),        %%xmm4 \n\t"  //       C30)
    //"mulpd  %%xmm0,              %%xmm11\n\t"  // scale ab_21_30 by alpha
    "mulpd  %%xmm1,              %%xmm4 \n\t"  // scale (C21, C30) by beta
    "minpd  %%xmm11,             %%xmm4 \n\t"  // add results
    "movlpd %%xmm4,        (%%r11)      \n\t"  // write back (C21,
    "movhpd %%xmm4,        (%%rdx,%%r8) \n\t"  //             C30)
    "                                   \n\t"
    "                                   \n\t"
    "leaq   (%%rcx,%%r9,2),      %%rcx  \n\t"  // Store addr of C02 in %rcx
    "leaq   (%%r10,%%r9,2),      %%r10  \n\t"  // Store addr of C03 in %r10
    "leaq   (%%rdx,%%r9,2),      %%rdx  \n\t"  // Store addr of C22 in $rdx
    "leaq   (%%r11,%%r9,2),      %%r11  \n\t"  // Store addr of C23 in %r11
    "                                   \n\t"
    "                                   \n\t"
    "movlpd (%%rcx),             %%xmm3 \n\t"  // load (C02,
    "movhpd (%%r10,%%r8),        %%xmm3 \n\t"  //       C13)
    //"mulpd  %%xmm0,              %%xmm12\n\t"  // scale ab_02_13 by alpha
    "mulpd  %%xmm1,              %%xmm3 \n\t"  // scale (C02, C13) by beta
    "minpd  %%xmm12,             %%xmm3 \n\t"  // add results
    "movlpd %%xmm3,        (%%rcx)      \n\t"  // write back (C02,
    "movhpd %%xmm3,        (%%r10,%%r8) \n\t"  //             C13)
    "                                   \n\t"
    "movlpd (%%rdx),             %%xmm4 \n\t"  // load (C22,
    "movhpd (%%r11, %%r8),       %%xmm4 \n\t"  //       C33)
    //"mulpd  %%xmm0,              %%xmm13\n\t"  // scale ab_22_33 by alpha
    "mulpd  %%xmm1,              %%xmm4 \n\t"  // scale (C22, C33) by beta
    "minpd  %%xmm13,             %%xmm4 \n\t"  // add results
    "movlpd %%xmm4,             (%%rdx) \n\t"  // write back (C22,
    "movhpd %%xmm4,        (%%r11,%%r8) \n\t"  //             C33)
    "                                   \n\t"
    "                                   \n\t"
    "movlpd (%%r10),             %%xmm3 \n\t"  // load (C03,
    "movhpd (%%rcx,%%r8),        %%xmm3 \n\t"  //       C12)
    //"mulpd  %%xmm0,              %%xmm14\n\t"  // scale ab_03_12 by alpha
    "mulpd  %%xmm1,              %%xmm3 \n\t"  // scale (C03, C12) by beta
    "minpd  %%xmm14,             %%xmm3 \n\t"  // add results
    "movlpd %%xmm3,        (%%r10)      \n\t"  // write back (C03,
    "movhpd %%xmm3,        (%%rcx,%%r8) \n\t"  //             C12)
    "                                   \n\t"
    "movlpd (%%r11),             %%xmm4 \n\t"  // load (C23,
    "movhpd (%%rdx,%%r8),        %%xmm4 \n\t"  //       C32)
    //"mulpd  %%xmm0,              %%xmm15\n\t"  // scale ab_23_32 by alpha
    "mulpd  %%xmm1,              %%xmm4 \n\t"  // scale (C23, C32) by beta
    "minpd  %%xmm15,             %%xmm4 \n\t"  // add results
    "movlpd %%xmm4,        (%%r11)      \n\t"  // write back (C23,
    "movhpd %%xmm4,        (%%rdx,%%r8) \n\t"  //             C32)
    : // output
    : // input
        "m" (kb),       // 0
        "m" (kl),       // 1
        "m" (A),        // 2
        "m" (B),        // 3
        "m" (alpha),    // 4
        "m" (beta),     // 5
        "m" (C),        // 6
        "m" (incRowC),  // 7
        "m" (incColC),  // 8
        "m" (nextA),    // 9
        "m" (nextB)     // 10
    : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15"
    );
}
