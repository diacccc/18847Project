#include "gemm_avx.h"
#include <unordered_map>
#include <functional>

#ifdef __APPLE__
  #include <arm_neon.h>
#else
  #include <immintrin.h>
#endif

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#define M_BLOCKING 96
#define N_BLOCKING 256
#define K_BLOCKING 192

namespace gemm {

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

    float *packed_A = new float[M * K];
    float *packed_B = new float[K * N];
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            packing_B_8x4_neon(
                &B.at(k_count,n_count), k_inc, n_inc, B.ld(),
                packed_B
            );
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                packing_A_8x4_neon(
                    &A.at(m_count,k_count), m_inc, k_inc, A.ld(),
                    packed_A
                );
                #ifdef __APPLE__
                    macro_kernel_8x4_sgemm_neon(
                        m_inc, n_inc, k_inc, 
                        alpha,
                        packed_A, A.ld(),
                        packed_B, B.ld(),
                        beta,
                        &C.at(m_count, n_count), C.ld()
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
    }
    
}

}