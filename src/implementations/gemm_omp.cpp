/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "gemm_omp.h"
#include <cassert>
#include <iostream>
#include <omp.h>
#include <sys/sysctl.h> // For sysctlbyname on macOS

namespace gemm
{

// Helper for packing sub-blocks of A (column-major)
void pack_micro_A(const float *A, size_t m, size_t k, size_t LDA, float *packed)
{
    // Pack A in a way optimized for the micro-kernel access pattern
    for (size_t j = 0; j < k; j++)
    {
        for (size_t i = 0; i < m; i++)
        {
            packed[i + j * m] = A[i + j * LDA];
        }
    }
}

// Helper for packing sub-blocks of B (column-major)
void pack_micro_B(const float *B, size_t k, size_t n, size_t LDB, float *packed)
{
    // Pack B for efficient access in the micro-kernel
    for (size_t j = 0; j < n; j++)
    {
        for (size_t i = 0; i < k; i++)
        {
            packed[i + j * k] = B[i + j * LDB];
        }
    }
}

// Get L3 cache size (platform-specific)
size_t get_l3_cache_size()
{
// Platform-specific code to detect L3 cache size
// For macOS:
#ifdef __APPLE__
    size_t size;
    size_t l3_size;
    size = sizeof(l3_size);
    if (sysctlbyname("hw.l3cachesize", &l3_size, &size, NULL, 0) == 0)
        printf("L3 cache size: %zu bytes\n", l3_size);
    return l3_size;
#endif

    // Default size (8MB is common for many systems)
    return 8 * 1024 * 1024;
}

void scale_matrix(float beta, Matrix<float> &C)
{
    if (beta == 0.0f)
    {
        C.fill(0.0f);
    }
    else if (beta != 1.0f)
    {
        const size_t M = C.rows();
        const size_t N = C.cols();
        const size_t LDC = C.ld();

#pragma omp parallel for collapse(2)
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t i = 0; i < M; ++i)
            {
                C.at(i, j) *= beta;
            }
        }
    }
}

void GemmOMP::micro_kernel_8x8(size_t K, float alpha, const float *A, const float *B, float *C, size_t LDC)
{
    float c[8][8] = {0};

    // Correct memory access pattern matching the packing functions
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < 8; i++)
        {
            const float a_val = A[i + k * 8];
            for (size_t j = 0; j < 8; j++)
            {
                c[i][j] += a_val * B[j * K + k];
            }
        }
    }

    // Store results
    for (size_t j = 0; j < 8; j++)
    {
        for (size_t i = 0; i < 8; i++)
        {
            C[i + j * LDC] += alpha * c[i][j];
        }
    }
}

void GemmOMP::micro_kernel_4x4(size_t M, size_t N, size_t K, float alpha, const float *A, const float *B, float *C,
                               size_t LDC)
{
    float c[4][4] = {0};

    // Corrected access pattern
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M; i++)
        {
            const float a_val = A[i + k * M];
            for (size_t j = 0; j < N; j++)
            {
                c[i][j] += a_val * B[j * K + k];
            }
        }
    }

    // Store results
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < M; i++)
        {
            C[i + j * LDC] += alpha * c[i][j];
        }
    }
}

void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C) {
    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    assert(B.rows() == K && C.rows() == M && C.cols() == N);

    // Scale C by beta
    scale_matrix(beta, C);

    // Define block sizes for cache efficiency
    const size_t BM = 64; // Block size for M dimension
    const size_t BN = 64; // Block size for N dimension
    const size_t BK = 64; // Block size for K dimension

    // Blocked matrix multiplication with OpenMP
    #pragma omp parallel for collapse(2)
    for (size_t jb = 0; jb < N; jb += BN) {
        for (size_t ib = 0; ib < M; ib += BM) {
            // Define actual block sizes (handle edge cases)
            size_t jb_end = std::min(jb + BN, N);
            size_t ib_end = std::min(ib + BM, M);

            // Allocate thread-local C block
            float C_local[BM][BN] = {0};

            // For each block of K
            for (size_t kb = 0; kb < K; kb += BK) {
                const size_t kb_end = std::min(kb + BK, K);

                // Process this block
                for (size_t j = jb; j < jb_end; ++j) {
                    for (size_t i = ib; i < ib_end; ++i) {
                        float sum = 0.0f;
                        for (size_t k = kb; k < kb_end; ++k) {
                            sum += A.at(i, k) * B.at(k, j);
                        }
                        C_local[i-ib][j-jb] += sum;
                    }
                }
            }

            // Update the global C with the local results
            for (size_t j = jb; j < jb_end; ++j) {
                for (size_t i = ib; i < ib_end; ++i) {
                    C.at(i, j) += alpha * C_local[i-ib][j-jb];
                }
            }
        }
    }
}

} // namespace gemm