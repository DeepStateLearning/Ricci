#ifndef gen_mul_h__
#define gen_mul_h__

#ifdef __cplusplus
extern "C" {
#endif

#if defined __FAST_MATH__
extern void dgemm_pure(int n, const double* A, double* C);
#else
extern void dgemm_nn(int n, const double* A, double* C);
#endif

#ifdef __cplusplus
}
#endif

#endif
