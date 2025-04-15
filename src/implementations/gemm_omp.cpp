/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "gemm_omp.h"
#include <cassert>
#include <iostream>
#include <omp.h>

namespace gemm
{

void GemmOMP::macro_kernel_4x4_sgemm(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA, const float *B,
                                     int LDB,
                                     float beta, // Not used directly as it's applied in execute()
                                     float *C, int LDC)
{
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i += 4)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            // Handle matrix boundaries
            const size_t m_block = std::min(4UL, M - i);
            const size_t n_block = std::min(4UL, N - j);

            // Initialize accumulation array
            float c[4][4] = {{0}};

            // Compute matrix multiplication for this block
            for (size_t k = 0; k < K; ++k)
            {
                for (size_t ii = 0; ii < m_block; ++ii)
                {
                    for (size_t jj = 0; jj < n_block; ++jj)
                    {
                        c[ii][jj] += alpha * A[(i + ii) * LDA + k] * B[k * LDB + (j + jj)];
                    }
                }
            }

            // Add results to output matrix (beta already applied in execute())
            for (size_t ii = 0; ii < m_block; ++ii)
            {
                for (size_t jj = 0; jj < n_block; ++jj)
                {
                    C[(i + ii) * LDC + (j + jj)] += c[ii][jj];
                }
            }
        }
    }
}

void GemmOMP::macro_kernel_8x8_sgemm(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA, const float *B,
                                     int LDB, float beta, float *C, int LDC)
{
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i += 4)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            // Handle matrix boundaries
            const size_t m_block = std::min(8UL, M - i);
            const size_t n_block = std::min(8UL, N - j);

            // Initialize accumulation array
            float c[8][8] = {{0}};

            // Compute matrix multiplication for this block
            for (size_t k = 0; k < K; ++k)
            {
                for (size_t ii = 0; ii < m_block; ++ii)
                {
                    for (size_t jj = 0; jj < n_block; ++jj)
                    {
                        c[ii][jj] += alpha * A[(i + ii) * LDA + k] * B[k * LDB + (j + jj)];
                    }
                }
            }

            // Add results to output matrix (beta already applied in execute())
            for (size_t ii = 0; ii < m_block; ++ii)
            {
                for (size_t jj = 0; jj < n_block; ++jj)
                {
                    C[(i + ii) * LDC + (j + jj)] += c[ii][jj];
                }
            }
        }
    }
}

void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
{
    const size_t K = A.cols();
    const size_t M = A.rows();
    const size_t N = B.cols();

    assert(B.rows() == K && C.rows() == M && C.cols() == N);
    scale(beta, C);

    // Choose kernel size based on matrix dimensions
    // For larger matrices, larger tiles might be beneficial
    if (M >= 16 && N >= 16 && K >= 16)
    {
        macro_kernel_8x8_sgemm(M, N, K, alpha, A.data(), A.ld(), B.data(), B.ld(), beta, C.data(), C.ld());
    }
    else if (M >= 8 && N >= 8)
    {
        macro_kernel_4x4_sgemm(M, N, K, alpha, A.data(), A.ld(), B.data(), B.ld(), beta, C.data(), C.ld());
    }
}
} // namespace gemm
