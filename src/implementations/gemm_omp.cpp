/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "gemm_omp.h"
#include <cassert>
#include <iostream>
#include <omp.h>
// #include <sys/sysctl.h>  // For sysctlbyname on macOS

namespace gemm
{
// Helper for packing sub-blocks of A (row-major)
void pack_micro_A(const float *A, size_t m, size_t k, size_t LDA, float *packed) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            packed[i * k + j] = A[i * LDA + j];
        }
    }
}

// Helper for packing sub-blocks of B (row-major)
void pack_micro_B(const float *B, size_t k, size_t n, size_t LDB, float *packed) {
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            packed[i * n + j] = B[i * LDB + j];
        }
    }
}

// Get L3 cache size (platform-specific)
size_t get_l3_cache_size() {
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
void GemmOMP::micro_kernel_8x8(size_t K, float alpha,
                              const float *A, const float *B,
                              float *C, size_t LDC) {
    float c[8][8] = {0};

    // For row-major layout
    for (size_t i = 0; i < 8; i++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < 8; j++) {
                c[i][j] += A[i * K + k] * B[k * 8 + j];
            }
        }
    }

    // Store results in row-major order
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 8; j++) {
            C[i * LDC + j] += alpha * c[i][j];
        }
    }
}

void GemmOMP::micro_kernel_4x4(size_t M, size_t N, size_t K, float alpha,
                              const float *A, const float *B,
                              float *C, size_t LDC) {
    float c[4][4] = {0};

    // For row-major layout
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < N; j++) {
                c[i][j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    // Store results in row-major order
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * LDC + j] += alpha * c[i][j];
        }
    }
}
void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C) {
    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    assert(B.rows() == K && C.rows() == M && C.cols() == N);

    // Handle beta scaling
    if (beta != 1.0f) {
        scale_matrix(beta, C);
    }

    const size_t LDA = A.ld();
    const size_t LDB = B.ld();
    const size_t LDC = C.ld();

    // Determine cache-based blocking parameters
    size_t L3_cache_size = get_l3_cache_size();

    // Set blocking parameters based on matrix sizes and cache
    size_t MC = 128;  // Block size for M dimension
    size_t NC = 128;  // Block size for N dimension
    size_t KC = 64;   // Block size for K dimension

    if (L3_cache_size > 0) {
        // Calculate MC to fit in cache
        size_t available_cache = L3_cache_size * 3 / 4;  // Use 75% of L3
        MC = (available_cache / 2) / (KC * sizeof(float));
        MC = (MC / 16) * 16;  // Round to multiple of 16 for alignment

        // Adjust if needed
        if (MC < 64) MC = 64;
        if (MC > 256) MC = 256;
    }

    // Allocate packed buffers once
    float *packed_A = new float[MC * KC * omp_get_max_threads()];
    float *packed_B = new float[KC * NC * omp_get_max_threads()];

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        float *thread_packed_A = packed_A + thread_id * (MC * KC);
        float *thread_packed_B = packed_B + thread_id * (KC * NC);

        // Main blocking loops
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < M; i += MC) {
            size_t m_block = std::min(MC, M - i);

            for (size_t k = 0; k < K; k += KC) {
                size_t k_block = std::min(KC, K - k);

                // Pack block of A
                pack_micro_A(&A(i, k), m_block, k_block, LDA, thread_packed_A);

                for (size_t j = 0; j < N; j += NC) {
                    size_t n_block = std::min(NC, N - j);

                    // Pack block of B
                    pack_micro_B(&B(k, j), k_block, n_block, LDB, thread_packed_B);

                    // Process micro-kernels
                    for (size_t ii = 0; ii < m_block; ii += 8) {
                        size_t micro_m = std::min(size_t(8), m_block - ii);

                        for (size_t jj = 0; jj < n_block; jj += 8) {
                            size_t micro_n = std::min(size_t(8), n_block - jj);

                            if (micro_m == 8 && micro_n == 8) {
                                // Full 8x8 kernel
                                micro_kernel_8x8(k_block, alpha,
                                               thread_packed_A + ii * k_block,
                                               thread_packed_B + jj,
                                               &C(i + ii, j + jj), LDC);
                            } else {
                                // Partial block kernel
                                micro_kernel_4x4(micro_m, micro_n, k_block, alpha,
                                               thread_packed_A + ii * k_block,
                                               thread_packed_B + jj,
                                               &C(i + ii, j + jj), LDC);
                            }
                        }
                    }
                }
            }
        }
    }

    // Free packed buffers
    delete[] packed_A;
    delete[] packed_B;
}
