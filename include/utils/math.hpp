//
// Created by ksnzh on 17-7-18.
//

#ifndef BLOBFLOW_MATH_HPP
#define BLOBFLOW_MATH_HPP

extern "C"{
#include <cblas.h>
}

#include <cmath>
#include "common.hpp"
#include "utils/math_alternative.hpp"

template<typename Dtype>
void bf_axpy(int N, Dtype alpha,const Dtype *x, Dtype *y);

template<typename Dtype>
void bf_cpu_axpby(int N, Dtype alpha, const Dtype *x, Dtype beta,Dtype *y);

//	C=alpha*A*B+beta*C
template<typename Dtype>
void bf_cpu_gemm(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
                     const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B,
                     const Dtype beta, Dtype *C);

//	y=alpha*A*x+beta*y
template<typename Dtype>
void bf_cpu_gemv(const CBLAS_TRANSPOSE transA, const int M, const int N, const Dtype alpha,
                     const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

template<typename Dtype>
Dtype bf_cpu_asum(int N, const Dtype *x);

template<typename Dtype>
void bf_copy(const int N, Dtype *dest, const Dtype *src);

template <typename Dtype>
void bf_set(const int N, const Dtype val, Dtype *x);

template <typename Dtype>
void bf_rng_uniform(const int N, const Dtype lower, const Dtype upper, Dtype *x);

template <typename Dtype>
void bf_rng_gaussian(const int N, const Dtype mu, const Dtype sigma, Dtype* x);

template <typename Dtype>
void bf_rng_bernoulli(const int N, const Dtype p, unsigned int* x);

template <typename Dtype>
void bf_exp(const int N, const Dtype* x, Dtype* y);

template <typename Dtype>
void bf_div(const int N, const Dtype* a, const Dtype* b,Dtype* y);

template <typename Dtype>
void bf_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
Dtype bf_cpu_strided_dot(const int N, const Dtype* x, const int incx,const Dtype* y, const int incy);

template <typename Dtype>
Dtype bf_cpu_dot(const int N, const Dtype* x, const Dtype* y);

template <typename Dtype>
void bf_scal(const int N, const Dtype alpha, Dtype* x);

template <typename Dtype>
void bf_scale(const int N, const Dtype alpha,const Dtype* x,Dtype* y);

template <typename Dtype>
void bf_powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void bf_add(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void bf_sub(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void bf_add_scalar(const int N, Dtype scalar,Dtype* y);

inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
    cblas_sscal(N, beta, Y, incY);
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
    cblas_dscal(N, beta, Y, incY);
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif //BLOBFLOW_MATH_HPP