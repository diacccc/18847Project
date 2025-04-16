/**
 * @file OMP implementation of GEMM
 * @author Molly Xiao minxiao@andrew.cmu.edu
 * @created 2025-04-12
 */
#include "gemm_omp.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>
#include <sys/sysctl.h> // For sysctlbyname on macOS

namespace gemm
{

// Pack A for better cache locality (column-major)
void pack_block_A(const Matrix<float> &A, float *packed, size_t ib, size_t kb, size_t block_m, size_t block_k)
{
    const size_t M = std::min(block_m, A.rows() - ib);
    const size_t K = std::min(block_k, A.cols() - kb);

	// Pack A in column-major order
	#pragma omp for collapse(2)
    for (size_t k = 0; k < K; k++)
    {
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

	#pragma omp for collapse(2)
    for (size_t n = 0; n < N; n++)
    {
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

	#pragma omp simd
    for (size_t k = 0; k < K; k++)
    {
        // Load A values (8 values from one column of A)
        float a0 = A[0 + k * 8];
        float a1 = A[1 + k * 8];
        float a2 = A[2 + k * 8];
        float a3 = A[3 + k * 8];
        float a4 = A[4 + k * 8];
        float a5 = A[5 + k * 8];
        float a6 = A[6 + k * 8];
        float a7 = A[7 + k * 8];

        // Load B values (8 values from one row of B)
        float b0 = B[k + 0 * K];
        float b1 = B[k + 1 * K];
        float b2 = B[k + 2 * K];
        float b3 = B[k + 3 * K];
        float b4 = B[k + 4 * K];
        float b5 = B[k + 5 * K];
        float b6 = B[k + 6 * K];
        float b7 = B[k + 7 * K];

        // Update C registers
        c[0][0] += a0 * b0;
        c[0][1] += a0 * b1;
        c[0][2] += a0 * b2;
        c[0][3] += a0 * b3;
        c[0][4] += a0 * b4;
        c[0][5] += a0 * b5;
        c[0][6] += a0 * b6;
        c[0][7] += a0 * b7;

        c[1][0] += a1 * b0;
        c[1][1] += a1 * b1;
        c[1][2] += a1 * b2;
        c[1][3] += a1 * b3;
        c[1][4] += a1 * b4;
        c[1][5] += a1 * b5;
        c[1][6] += a1 * b6;
        c[1][7] += a1 * b7;

        c[2][0] += a2 * b0;
        c[2][1] += a2 * b1;
        c[2][2] += a2 * b2;
        c[2][3] += a2 * b3;
        c[2][4] += a2 * b4;
        c[2][5] += a2 * b5;
        c[2][6] += a2 * b6;
        c[2][7] += a2 * b7;

        c[3][0] += a3 * b0;
        c[3][1] += a3 * b1;
        c[3][2] += a3 * b2;
        c[3][3] += a3 * b3;
        c[3][4] += a3 * b4;
        c[3][5] += a3 * b5;
        c[3][6] += a3 * b6;
        c[3][7] += a3 * b7;

        c[4][0] += a4 * b0;
        c[4][1] += a4 * b1;
        c[4][2] += a4 * b2;
        c[4][3] += a4 * b3;
        c[4][4] += a4 * b4;
        c[4][5] += a4 * b5;
        c[4][6] += a4 * b6;
        c[4][7] += a4 * b7;

        c[5][0] += a5 * b0;
        c[5][1] += a5 * b1;
        c[5][2] += a5 * b2;
        c[5][3] += a5 * b3;
        c[5][4] += a5 * b4;
        c[5][5] += a5 * b5;
        c[5][6] += a5 * b6;
        c[5][7] += a5 * b7;

        c[6][0] += a6 * b0;
        c[6][1] += a6 * b1;
        c[6][2] += a6 * b2;
        c[6][3] += a6 * b3;
        c[6][4] += a6 * b4;
        c[6][5] += a6 * b5;
        c[6][6] += a6 * b6;
        c[6][7] += a6 * b7;

        c[7][0] += a7 * b0;
        c[7][1] += a7 * b1;
        c[7][2] += a7 * b2;
        c[7][3] += a7 * b3;
        c[7][4] += a7 * b4;
        c[7][5] += a7 * b5;
        c[7][6] += a7 * b6;
        c[7][7] += a7 * b7;
    }

	// Update C matrix with the final results using 2D array notation
	#pragma omp parallel for collapse(2)
    for (int i = 0; i < 8; i++)
    {
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

	#pragma omp parallel for collapse(2) private(a_val)
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

	#pragma omp parallel for collapse(2)
    for (size_t j = 0; j < N; j++)
    {
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
        C.fill(0.0f);
    }
    else if (beta != 1.0f)
    {
        const size_t M = C.rows();
        const size_t N = C.cols();

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

void GemmOMP::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
{
    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    assert(B.rows() == K && C.rows() == M && C.cols() == N);

    // Scale C by beta
    scale_matrix(beta, C);

    // Define cache-efficient block sizes
    const size_t MC = 120; // Block size for M dimension
    const size_t KC = 240; // Block size for K dimension
    const size_t NC = 960; // Block size for N dimension

    // Micro-kernel tile sizes
    const size_t MR = 8; // 8x8 micro-kernel
    const size_t NR = 8;

// Allocate packed memory blocks (one per thread)
#pragma omp parallel
    {
        // Thread-local storage for packed blocks
        float *packed_A = new float[MC * KC];
        float *packed_B = new float[KC * NC];

// 3-level nested blocking for cache efficiency
#pragma omp for schedule(dynamic)
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

        delete[] packed_A;
        delete[] packed_B;
    }
}

} // namespace gemm