#ifndef __APPLE__
#include "gemm_simd.h"
#include <unordered_map>
#include <functional>
#include <arm_neon.h>

#define A(i, j) A[(i)+(j)*LDA]
#define B(i, j) B[(i)+(j)*LDB]
#define C(i, j) C[(i)+(j)*LDC]

namespace gemm {

void GemmSIMD::macro_kernel_4x1_sgemm_intel(
    size_t M, size_t N, size_t K, 
    float alpha, 
    const float *A, int LDA, 
    const float *B, int LDB, 
    float beta, 
    float *C, int LDC
) {
    #ifndef __APPLE__
    __m128 valpha = _mm_set1_ps(alpha);
    for (size_t i = 0; i < M; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                __m128 a = _mm_mul_ps(valpha, _mm_loadu_ps(&A(i, k)));
                __m128 b0 = _mm_set1_ps(B(k, j));
                __m128 b1 = _mm_set1_ps(B(k, j + 1));
                __m128 b2 = _mm_set1_ps(B(k, j + 2));
                __m128 b3 = _mm_set1_ps(B(k, j + 3));
                
                // Apple Silicon has no fma instruction
                // c0 = _mm_fmadd_ps(a, b0, c0);
                // c1 = _mm_fmadd_ps(a, b1, c1);
                // c2 = _mm_fmadd_ps(a, b2, c2);
                // c3 = _mm_fmadd_ps(a, b3, c3);
                c0 = _mm_add_ps(_mm_mul_ps(a, b0), c0);
                c1 = _mm_add_ps(_mm_mul_ps(a, b1), c1);
                c2 = _mm_add_ps(_mm_mul_ps(a, b2), c2);
                c3 = _mm_add_ps(_mm_mul_ps(a, b3), c3);
            }
            _mm_storeu_ps(&C(i, j    ), _mm_add_ps(c0, _mm_loadu_ps(&C(i, j   ))));
            _mm_storeu_ps(&C(i, j + 1), _mm_add_ps(c1, _mm_loadu_ps(&C(i, j + 1))));
            _mm_storeu_ps(&C(i, j + 2), _mm_add_ps(c2, _mm_loadu_ps(&C(i, j + 2))));
            _mm_storeu_ps(&C(i, j + 3), _mm_add_ps(c3, _mm_loadu_ps(&C(i, j + 3))));
        }
    }
    #endif
}


}

#endif