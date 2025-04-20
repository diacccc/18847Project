/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "../../include/gemm_omp.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sys/sysctl.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace gemm
{

    // Pack A for better cache locality (column-major)
    void pack_block_A(const Matrix<float> &A, float *packed, size_t ib, size_t kb, size_t block_m, size_t block_k)
    {
        const size_t M = std::min(block_m, A.rows() - ib);
        const size_t K = std::min(block_k, A.cols() - kb);

	    // Pack A in column-major order
        for (size_t k = 0; k < K; k++)
        {
            #pragma omp simd
            for (size_t m = 0; m < M; m++)
            {
                packed[m + k * M] = A.at(ib + m, kb + k);
            }
        }
    }

    // Pack B for better cache locality (row-major for faster inner loop)
    void pack_block_B(const Matrix<float> &B, float *packed, size_t kb, size_t jb, size_t block_k, size_t block_n)
    {
        const size_t K = std::min(block_k, B.rows() - kb);
        const size_t N = std::min(block_n, B.cols() - jb);

        for (size_t n = 0; n < N; n++)
        {
            #pragma omp simd
            for (size_t k = 0; k < K; k++)
            {
                packed[k + n * K] = B.at(kb + k, jb + n);
            }
        }
    }

    // Micro kernel for 8x8 blocks - optimized for vectorization
    void GemmOMP::micro_kernel_8x8(size_t K, float alpha, const float *A, const float *B, float *C, size_t LDC)
    {
        float c[8][8] = {0};

        for (size_t k = 0; k < K; ++k)
        {
            float a[8];
            float b[8];

		    #pragma omp simd
            for (int i = 0; i < 8; ++i)
                a[i] = A[i + k * 8];     // A is 8 x K, column-major or panel-packed

            #pragma omp simd
            for (int j = 0; j < 8; ++j)
                b[j] = B[k + j * K];     // B is K x 8, column-major

            for (int i = 0; i < 8; ++i)
            {
                #pragma omp simd
                for (int j = 0; j < 8; ++j)
                {
                    c[i][j] += a[i] * b[j];
                }
            }
        }

	    // Update C matrix with the final results using 2D array notation
        for (int i = 0; i < 8; i++)
        {
            #pragma omp simd
            for (int j = 0; j < 8; j++)
            {
                C[i + j * LDC] += alpha * c[i][j];
            }
        }
    }

    // Handle smaller blocks with simplified kernel
    void GemmOMP::micro_kernel_4x4(size_t M, size_t N, size_t K, float alpha, const float *A, const float *B, float *C,
                                size_t LDC)
    {
        float c[4][4] = {0};

        for (size_t k = 0; k < K; k++)
        {
            for (size_t i = 0; i < M; i++)
            {
                float a_val = A[i + k * M];
                for (size_t j = 0; j < N; j++)
                {
                    c[i][j] += a_val * B[k + j * K];
                }
            }
        }

        for (size_t j = 0; j < N; j++)
        {
            #pragma omp simd
            for (size_t i = 0; i < M; i++)
            {
                C[i + j * LDC] += alpha * c[i][j];
            }
        }
    }

    void scale_matrix(float beta, Matrix<float> &C)
    {
        if (beta == 0.0f)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < C.rows(); ++i) {
                for (size_t j = 0; j < C.cols(); ++j) {
                    C.at(i, j) = 0.0f;
                }
            }
        }
        else if (beta != 1.0f)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < C.rows(); ++i) {
                for (size_t j = 0; j < C.cols(); ++j) {
                    C.at(i, j) *= beta;
                }
            }
        }
    }

    void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
    {
        const size_t M = A.rows();
        const size_t K = A.cols();
        const size_t N = B.cols();

        assert(B.rows() == K && C.rows() == M && C.cols() == N);

        // Scale C by beta
        scale_matrix(beta, C);

        // Define cache-efficient block sizes
        const size_t MC = 240; // Block size for M dimension
        const size_t KC = 512; // Block size for K dimension
        const size_t NC = 2048; // Block size for N dimension

        // Micro-kernel tile sizes
        const size_t MR = 8; // 8x8 micro-kernel
        const size_t NR = 8;

        #pragma omp parallel
        {
            // Thread-local storage for packed blocks
            float *packed_A = (float *)aligned_alloc(64, MC * KC * sizeof(float));
            float *packed_B = (float *)aligned_alloc(64, KC * NC * sizeof(float));


            // Dynamic scheduling with a smaller chunk size help load balancing
            #pragma omp for
            for (size_t jc = 0; jc < N; jc += NC)
            {
                size_t nc = std::min(N - jc, NC);

                for (size_t pc = 0; pc < K; pc += KC)
                {
                    size_t kc = std::min(K - pc, KC);

                    // Pack B block (shared among iterations over ic)
                    for (size_t jr = 0; jr < nc; jr += NR)
                    {
                        size_t nr = std::min(nc - jr, NR);
                        if (nr == NR)
                        {
                            pack_block_B(B, packed_B + jr * kc, pc, jc + jr, kc, NR);
                        }
                    }

                    for (size_t ic = 0; ic < M; ic += MC)
                    {
                        size_t mc = std::min(M - ic, MC);

                        // Pack A block
                        for (size_t ir = 0; ir < mc; ir += MR)
                        {
                            size_t mr = std::min(mc - ir, MR);
                            if (mr == MR)
                            {
                                pack_block_A(A, packed_A + ir * kc, ic + ir, pc, MR, kc);
                            }
                        }

                        // Compute with packed data
                        for (size_t jr = 0; jr < nc; jr += NR)
                        {
                            size_t nr = std::min(nc - jr, NR);

                            for (size_t ir = 0; ir < mc; ir += MR)
                            {
                                size_t mr = std::min(mc - ir, MR);

                                if (mr == MR && nr == NR)
                                {
                                    // Full tile - use optimized kernel
                                    micro_kernel_8x8(kc, alpha, packed_A + ir * kc, packed_B + jr * kc,
                                                     &C.at(ic + ir, jc + jr), C.ld());
                                }
                                else
                                {
                                    // Partial tile - use generic kernel
                                    micro_kernel_4x4(mr, nr, kc, alpha, packed_A + ir * kc, packed_B + jr * kc,
                                                     &C.at(ic + ir, jc + jr), C.ld());
                                }
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