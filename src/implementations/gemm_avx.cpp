#include "gemm_avx.h"
#include <unordered_map>
#include <functional>

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#ifdef __APPLE__
//   #define SIMDE_ENABLE_NATIVE_ALIASES
//   #include <simde/x86/avx2.h>
  #include <arm_neon.h>
#else
  #include <immintrin.h>
#endif
namespace gemm {

void GemmAVX::macro_kernel_4x1_sgemm_neon(
    size_t M, size_t N, size_t K, 
    float alpha, 
    const float *A, int LDA, 
    const float *B, int LDB, 
    float beta, 
    float *C, int LDC
) {
    #ifdef __APPLE__
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            for (size_t k = 0; k < K; ++k) {
                float32x4_t a = vmulq_f32(valpha, vld1q_f32(&A(i, k)));
                float32x4_t b0 = vdupq_n_f32(B(k, j));
                float32x4_t b1 = vdupq_n_f32(B(k, j + 1));
                float32x4_t b2 = vdupq_n_f32(B(k, j + 2));
                float32x4_t b3 = vdupq_n_f32(B(k, j + 3));
                
                // On Apple Silicon (ARM), we can use FMA
                #ifdef __ARM_FEATURE_FMA
                c0 = vfmaq_f32(c0, a, b0);
                c1 = vfmaq_f32(c1, a, b1);
                c2 = vfmaq_f32(c2, a, b2);
                c3 = vfmaq_f32(c3, a, b3);
                #else
                c0 = vaddq_f32(vmulq_f32(a, b0), c0);
                c1 = vaddq_f32(vmulq_f32(a, b1), c1);
                c2 = vaddq_f32(vmulq_f32(a, b2), c2);
                c3 = vaddq_f32(vmulq_f32(a, b3), c3);
                #endif
            }
            vst1q_f32(&C(i, j),     vaddq_f32(c0, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c1, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i, j + 2), vaddq_f32(c2, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c3, vld1q_f32(&C(i, j + 3))));
        }
    }
    #endif
}

void GemmAVX::macro_kernel_4x1_sgemm_intel(
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

// Implementation of NaiveCpuGemm::execute
void GemmAVX::execute(
    float alpha,
    const Matrix<float>& A,
    const Matrix<float>& B,
    float beta,
    Matrix<float>& C
) {
    const size_t M = A.rows();
    const size_t N = B.cols();
    const size_t K = A.cols();
    
    assert(B.rows() == K);
    assert(C.rows() == M);
    assert(C.cols() == N);
    
    scale(beta, C);
    #ifdef __APPLE__
        macro_kernel_4x1_sgemm_neon(
            M, N, K, 
            alpha,
            A.data(), A.ld(),
            B.data(), B.ld(),
            beta,
            C.data(), C.ld()
        );
    #else 
        macro_kernel_4x1_sgemm_intel(
            M, N, K, 
            alpha,
            A.data(), A.ld(),
            B.data(), B.ld(),
            beta,
            C.data(), C.ld()
        );
    #endif
}

}