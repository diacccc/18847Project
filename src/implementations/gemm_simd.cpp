#include "gemm_simd.h"

#include <functional>
#include <unordered_map>

#ifdef __APPLE__
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#define A(i, j) A[(i) + (j) * LDA]
#define B(i, j) B[(i) + (j) * LDB]
#define C(i, j) C[(i) + (j) * LDC]

#define M_BLOCKING 96
#define N_BLOCKING 256
#define K_BLOCKING 192

namespace gemm
{

// Implementation of NaiveCpuGemm::execute
void GemmSIMD::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
{
    const size_t M = A.rows();
    const size_t N = B.cols();
    const size_t K = A.cols();

    assert(B.rows() == K);
    assert(C.rows() == M);
    assert(C.cols() == N);

    scale(beta, C);

    float *packed_A = (float *)aligned_alloc(4096, K_BLOCKING * M_BLOCKING * sizeof(float));
    float *packed_B = (float *)aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(float));
    size_t m_count, n_count, k_count;
    size_t m_inc, n_inc, k_inc;
    for (n_count = 0; n_count < N; n_count += n_inc)
    {
        n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;
        for (k_count = 0; k_count < K; k_count += k_inc)
        {
            k_inc = (K - k_count > K_BLOCKING) ? K_BLOCKING : K - k_count;
            packing_B_8_neon(&B.at(k_count, n_count), k_inc, n_inc, B.ld(), packed_B);
            for (m_count = 0; m_count < M; m_count += m_inc)
            {
                m_inc = (M - m_count > M_BLOCKING) ? M_BLOCKING : N - m_count;
                packing_A_16_neon(&A.at(m_count, k_count), m_inc, k_inc, A.ld(), packed_A);
#ifdef __APPLE__
                macro_kernel_16x8_sgemm_neon(m_inc, n_inc, k_inc, alpha, packed_A, A.ld(), packed_B, B.ld(), beta,
                                             &C.at(m_count, n_count), C.ld());

#else
                macro_kernel_4x1_sgemm_intel(M, N, K, alpha, A.data(), A.ld(), B.data(), B.ld(), beta, C.data(),
                                             C.ld());
#endif
            }
        }
    }
}

} // namespace gemm