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

namespace gemm
{
    void pack_block_A(const Matrix<float> &A, float* packed, size_t ib, size_t kb, size_t M, size_t K)
    {
        // Get direct pointer to data for vectorization
        const float* A_data = A.data();
        const size_t LDA = A.ld();

        // Pack A for better cache locality - row-major order
        for (size_t m = 0; m < M; m++) {
            // Use direct pointer access for better vectorization
            #pragma omp simd
            for (size_t k = 0; k < K; k++) {
                packed[k + m * K] = A_data[(ib + m) + (kb + k) * LDA];
            }
        }
    }

    void pack_block_B(const Matrix<float> &B, float* packed, size_t kb, size_t jb, size_t K, size_t N)
    {
        // Get direct pointer to data for vectorization
        const float* B_data = B.data();
        const size_t LDB = B.ld();

        // Pack B for better cache locality - column-major order
        for (size_t n = 0; n < N; n++) {
            // Use direct pointer access for better vectorization
            #pragma omp simd
            for (size_t k = 0; k < K; k++) {
                packed[k + n * K] = B_data[(kb + k) + (jb + n) * LDB];
            }
        }
    }

    // SIMD-optimized matrix multiplication
    void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
    {
        const size_t M = A.rows();
        const size_t N = B.cols();
        const size_t K = A.cols();

        assert(B.rows() == K);
        assert(C.rows() == M);
        assert(C.cols() == N);

        size_t K_BLOCKING = 32;
        size_t M_BLOCKING = 64;
        size_t N_BLOCKING = 64;

        // Apply beta scaling to C
        scale(beta, C);

        // Get direct pointer to C data for the final update
        float* C_data = C.data();
        const size_t LDC = C.ld();

        #pragma omp parallel
        {
            // Each thread needs its own workspace
            float *packed_A = (float *) aligned_alloc(64, K_BLOCKING*M_BLOCKING*sizeof(float));
            float *packed_B = (float *) aligned_alloc(64, K_BLOCKING*N_BLOCKING*sizeof(float));

            #pragma omp for
            for (size_t i = 0; i < M; i += M_BLOCKING) {
                size_t m_size = std::min(M_BLOCKING, M - i);

                for (size_t k = 0; k < K; k += K_BLOCKING) {
                    size_t k_size = std::min(K_BLOCKING, K - k);
                    pack_block_A(A, packed_A, i, k, m_size, k_size);

                    for (size_t j = 0; j < N; j += N_BLOCKING) {
                        size_t n_size = std::min(N_BLOCKING, N - j);
                        pack_block_B(B, packed_B, k, j, k_size, n_size);

                        // Process the block with SIMD optimizations
                        for (size_t ii = 0; ii < m_size; ii++) {
                            for (size_t jj = 0; jj < n_size; jj++) {
                                float sum = 0.0f;

                                // This is where SIMD vectorization happens effectively
                                #pragma omp simd reduction(+:sum)
                                for (size_t kk = 0; kk < k_size; kk++) {
                                    sum += packed_A[kk + ii * k_size] * packed_B[kk + jj * k_size];
                                }

                                // Direct pointer access for better performance
                                C_data[(i + ii) + (j + jj) * LDC] += alpha * sum;
                            }
                        }
                    }
                }
            }

            free(packed_A);
            free(packed_B);
        }
    }

} // namespace gemm