/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "../../include/gemm_omp.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#ifdef _OPENMP
    #include <omp.h>
#endif

// Optimize block sizes for better cache utilization
#define M_BLOCKING 96
#define N_BLOCKING 192
#define K_BLOCKING 192

// Micro-kernel sizes for register blocking
#define MR 8
#define NR 8

namespace gemm
{
    // Optimized packing of A with better memory layout for vectorization
    void pack_block_A(const Matrix<float> &A, float* __restrict__ packed,
                    size_t ib, size_t kb, size_t M, size_t K)
    {
        const float* __restrict__ A_data = A.data();
        const size_t LDA = A.ld();

        // Pack A in panels that match the micro-kernel (MR x K)
        for (size_t mp = 0; mp < M; mp += MR) {
            const size_t m_min = M - mp > MR ? MR : M - mp;

            for (size_t k = 0; k < K; k++) {
                size_t kMR = k * MR;
                #pragma omp simd
                for (size_t m = 0; m < m_min; m++) {
                    // Packed format optimized for micro-kernel access pattern
                    packed[m + kMR] = A_data[(ib + mp + m) + (kb + k) * LDA];
                }

                // Pad with zeros if needed
                for (size_t m = m_min; m < MR; m++) {
                    packed[m + kMR] = 0.0f;
                }
            }
            packed += MR * K; // Move to the next panel
        }
    }

    // Optimized packing of B with better memory layout for vectorization
    void pack_block_B(const Matrix<float> &B, float* __restrict__ packed,
                    size_t kb, size_t jb, size_t K, size_t N)
    {
        const float* __restrict__ B_data = B.data();
        const size_t LDB = B.ld();

        // Pack B in panels that match the micro-kernel (K x NR)
        for (size_t np = 0; np < N; np += NR) {
            const size_t n_min = N - np > NR ? NR : N - np;

            for (size_t n = 0; n < n_min; n++) {
                size_t nK = n * K;
                #pragma omp simd
                for (size_t k = 0; k < K; k++) {
                    // Packed format optimized for micro-kernel access pattern
                    packed[k + nK] = B_data[(kb + k) + (jb + np + n) * LDB];
                }
            }

            // Pad with zeros if needed
            for (size_t n = n_min; n < NR; n++) {
                size_t nK = n * K;
                for (size_t k = 0; k < K; k++) {
                    packed[k + nK] = 0.0f;
                }
            }

            packed += K * NR; // Move to the next panel
        }
    }

    // Highly optimized micro-kernel for MR x NR blocks
    void micro_kernel(size_t K, float alpha,
                    const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C, size_t LDC)
    {
        // Local accumulators for better register reuse
        float c[MR][NR] = {{0}};

        // Main computation loop - this is the most critical part for performance
        for (size_t k = 0; k < K; k++) {
            size_t kMR = k * MR;
            // Load a single column of A (MR elements)
            float a0 = A[0 + kMR];
            float a1 = A[1 + kMR];
            float a2 = A[2 + kMR];
            float a3 = A[3 + kMR];
            float a4 = A[4 + kMR];
            float a5 = A[5 + kMR];
            float a6 = A[6 + kMR];
            float a7 = A[7 + kMR];

            // For each element in the column of A, multiply by row of B
            #pragma omp simd
            for (size_t j = 0; j < NR; j++) {
                float b_val = B[k + j * K];
                c[0][j] += a0 * b_val;
                c[1][j] += a1 * b_val;
                c[2][j] += a2 * b_val;
                c[3][j] += a3 * b_val;
                c[4][j] += a4 * b_val;
                c[5][j] += a5 * b_val;
                c[6][j] += a6 * b_val;
                c[7][j] += a7 * b_val;
            }
        }

        // Store results back to C with alpha scaling - fully unrolled
        #pragma omp simd
        for (size_t j = 0; j < NR; j++) {
            const size_t jLDC = j * LDC;
            C[0 + jLDC] += alpha * c[0][j];
            C[1 + jLDC] += alpha * c[1][j];
            C[2 + jLDC] += alpha * c[2][j];
            C[3 + jLDC] += alpha * c[3][j];
            C[4 + jLDC] += alpha * c[4][j];
            C[5 + jLDC] += alpha * c[5][j];
            C[6 + jLDC] += alpha * c[6][j];
            C[7 + jLDC] += alpha * c[7][j];
        }
    }

    // Highly optimized GEMM implementation
    void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
    {
        const size_t M = A.rows();
        const size_t N = B.cols();
        const size_t K = A.cols();

        assert(B.rows() == K);
        assert(C.rows() == M);
        assert(C.cols() == N);

        // Apply beta scaling to C
        scale(beta, C);

        // Get direct pointer to C data for the final update
        float* C_data = C.data();
        const size_t LDC = C.ld();

        #pragma omp parallel
        {
            // Each thread needs its own workspace - aligned for SIMD operations
            // Use a larger alignment (64 bytes) for better SIMD performance
            float* packed_A = (float*) aligned_alloc(64, M_BLOCKING * K_BLOCKING * sizeof(float));
            float* packed_B = (float*) aligned_alloc(64, K_BLOCKING * N_BLOCKING * sizeof(float));

            #pragma omp for schedule(dynamic)
            for (size_t n = 0; n < N; n += N_BLOCKING) {
                size_t n_size = N - n > N_BLOCKING ? N_BLOCKING : N - n;

                for (size_t k = 0; k < K; k += K_BLOCKING) {
                    size_t k_size = K - k > K_BLOCKING ? K_BLOCKING : K - k;

                    // Pack B block once and reuse for multiple A blocks
                    pack_block_B(B, packed_B, k, n, k_size, n_size);

                    for (size_t m = 0; m < M; m += M_BLOCKING) {
                        size_t m_size = M - m > M_BLOCKING ? M_BLOCKING : M - m;

                        // Pack A block
                        pack_block_A(A, packed_A, m, k, m_size, k_size);

                        // Process packed blocks with micro-kernels
                        for (size_t nn = 0; nn < n_size; nn += NR) {
                            size_t n_micro = n_size - nn > NR ? NR : n_size - nn;
                            size_t c_base_col = n + nn;

                            for (size_t mm = 0; mm < m_size; mm += MR) {
                                size_t m_micro = m_size - mm > MR ? MR : m_size - mm;
                                size_t c_base_row = m + mm;

                                // If we have a full micro-kernel block, use the optimized version
                                if (m_micro == MR && n_micro == NR) {
                                    // Point to the correct positions in packed data
                                    const float* a_micro = packed_A + mm * k_size;
                                    const float* b_micro = packed_B + nn * k_size;
                                    float* c_micro = C_data + (c_base_row) + (c_base_col) * LDC;

                                    // Execute micro-kernel
                                    micro_kernel(k_size, alpha, a_micro, b_micro, c_micro, LDC);
                                } else {
                                    // move the * ops to the outer loop to reduce overhead, reuse variables
                                    size_t a_base = mm * k_size;
                                    size_t b_base = nn * k_size;

                                    for (size_t mr = 0; mr < m_micro; mr++) {
                                        size_t a_offset = (a_base + mr * k_size);
                                        size_t c_row = c_base_row + mr;

                                        for (size_t nr = 0; nr < n_micro; nr++) {
                                            float sum = 0.0f;
                                            size_t b_offset = b_base + nr * k_size;

                                            #pragma omp simd reduction(+:sum)
                                            for (size_t kk = 0; kk < k_size; kk++) {
                                                sum += packed_A[a_offset + kk] * packed_B[b_offset + kk];
                                            }

                                            C_data[c_row + (c_base_col + nr) * LDC] += alpha * sum;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Cleanup thread-local memory
            free(packed_A);
            free(packed_B);
        }
    }

} // namespace gemm