#ifndef __APPLE__

#include "gemm_simd.h"
#include <functional>
#include <immintrin.h>
#include <unordered_map>

#define A(i, j) A[(i) + (j) * LDA]
#define B(i, j) B[(i) + (j) * LDB]
#define C(i, j) C[(i) + (j) * LDC]

namespace gemm
{

void GemmSIMD::packing_A_8_intel(const float *A, size_t M, size_t K, size_t LDA, float *packed_A)
{
    float *dst = packed_A;
    for (size_t i = 0; i < M; i += 8)
    {
        for (size_t j = 0; j < K; ++j)
        {
            __m256 block =
                _mm256_set_ps(A[(i + 7) + j * LDA], A[(i + 6) + j * LDA], A[(i + 5) + j * LDA], A[(i + 4) + j * LDA],
                              A[(i + 3) + j * LDA], A[(i + 2) + j * LDA], A[(i + 1) + j * LDA], A[(i + 0) + j * LDA]);
            _mm256_storeu_ps(dst, block);
            dst += 8;
        }
    }
}

void GemmSIMD::packing_B_8_intel(const float *B, size_t K, size_t N, size_t LDB, float *packed_B)
{
    float *dst = packed_B;
    for (size_t j = 0; j < N; j += 8)
    {
        const float *B_ptr0 = &B[0 + j * LDB];
        const float *B_ptr1 = &B[0 + (j + 1) * LDB];
        const float *B_ptr2 = &B[0 + (j + 2) * LDB];
        const float *B_ptr3 = &B[0 + (j + 3) * LDB];
        const float *B_ptr4 = &B[0 + (j + 4) * LDB];
        const float *B_ptr5 = &B[0 + (j + 5) * LDB];
        const float *B_ptr6 = &B[0 + (j + 6) * LDB];
        const float *B_ptr7 = &B[0 + (j + 7) * LDB];

        for (size_t k = 0; k < K; ++k)
        {
            dst[0] = *B_ptr0++;
            dst[1] = *B_ptr1++;
            dst[2] = *B_ptr2++;
            dst[3] = *B_ptr3++;
            dst[4] = *B_ptr4++;
            dst[5] = *B_ptr5++;
            dst[6] = *B_ptr6++;
            dst[7] = *B_ptr7++;
            dst += 8;
        }
    }
}

#define KERNEL_8x8_SGEMM_INTEL                                                                                         \
    a = _mm256_mul_ps(_mm256_loadu_ps(packed_A), valpha);                                                              \
    b0 = _mm256_set1_ps(packed_B[0]);                                                                                  \
    b1 = _mm256_set1_ps(packed_B[1]);                                                                                  \
    b2 = _mm256_set1_ps(packed_B[2]);                                                                                  \
    b3 = _mm256_set1_ps(packed_B[3]);                                                                                  \
    b4 = _mm256_set1_ps(packed_B[4]);                                                                                  \
    b5 = _mm256_set1_ps(packed_B[5]);                                                                                  \
    b6 = _mm256_set1_ps(packed_B[6]);                                                                                  \
    b7 = _mm256_set1_ps(packed_B[7]);                                                                                  \
    c0 = _mm256_fmadd_ps(a, b0, c0);                                                                                   \
    c1 = _mm256_fmadd_ps(a, b1, c1);                                                                                   \
    c2 = _mm256_fmadd_ps(a, b2, c2);                                                                                   \
    c3 = _mm256_fmadd_ps(a, b3, c3);                                                                                   \
    c4 = _mm256_fmadd_ps(a, b4, c4);                                                                                   \
    c5 = _mm256_fmadd_ps(a, b5, c5);                                                                                   \
    c6 = _mm256_fmadd_ps(a, b6, c6);                                                                                   \
    c7 = _mm256_fmadd_ps(a, b7, c7);                                                                                   \
    packed_A += 8;                                                                                                     \
    packed_B += 8;                                                                                                     \
    k++;
void GemmSIMD::macro_kernel_8x8_sgemm_intel(size_t M, size_t N, size_t K, float alpha, const float *A, int,
                                            const float *B, int, float, float *C, int LDC)
{
    const float *packed_A;
    const float *packed_B;

    for (size_t i = 0; i < M; i += 8)
    {
        for (size_t j = 0; j < N; j += 8)
        {
            __m256 a, b0, b1, b2, b3, b4, b5, b6, b7;
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();
            __m256 c6 = _mm256_setzero_ps();
            __m256 c7 = _mm256_setzero_ps();

            packed_A = A + i * K;
            packed_B = B + j * K;
            __m256 valpha = _mm256_set1_ps(alpha);

            for (size_t k = 0; k < K;)
            {
                KERNEL_8x8_SGEMM_INTEL KERNEL_8x8_SGEMM_INTEL KERNEL_8x8_SGEMM_INTEL KERNEL_8x8_SGEMM_INTEL
            }

            // 写回 C（假设 C 是列主存储）
            for (int col = 0; col < 8; ++col)
            {
                float *c_col = &C[i + (j + col) * LDC];
                __m256 c_orig = _mm256_loadu_ps(c_col);
                __m256 c_new = _mm256_add_ps(c_orig, col == 0   ? c0
                                                     : col == 1 ? c1
                                                     : col == 2 ? c2
                                                     : col == 3 ? c3
                                                     : col == 4 ? c4
                                                     : col == 5 ? c5
                                                     : col == 6 ? c6
                                                                : c7);
                _mm256_storeu_ps(c_col, c_new);
            }
        }
    }
}

void GemmSIMD::macro_kernel_4x4_sgemm_intel(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA,
                                            const float *B, int LDB, float beta, float *C, int LDC)
{
#ifndef __APPLE__
    __m128 valpha = _mm_set1_ps(alpha);
    for (size_t i = 0; i < M; i += 4)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            for (size_t k = 0; k < K; ++k)
            {
                __m128 a = _mm_mul_ps(valpha, _mm_loadu_ps(&A(i, k)));
                __m128 b0 = _mm_set1_ps(B(k, j));
                __m128 b1 = _mm_set1_ps(B(k, j + 1));
                __m128 b2 = _mm_set1_ps(B(k, j + 2));
                __m128 b3 = _mm_set1_ps(B(k, j + 3));

                c0 = _mm_fmadd_ps(a, b0, c0);
                c1 = _mm_fmadd_ps(a, b1, c1);
                c2 = _mm_fmadd_ps(a, b2, c2);
                c3 = _mm_fmadd_ps(a, b3, c3);
            }
            _mm_storeu_ps(&C(i, j), _mm_add_ps(c0, _mm_loadu_ps(&C(i, j))));
            _mm_storeu_ps(&C(i, j + 1), _mm_add_ps(c1, _mm_loadu_ps(&C(i, j + 1))));
            _mm_storeu_ps(&C(i, j + 2), _mm_add_ps(c2, _mm_loadu_ps(&C(i, j + 2))));
            _mm_storeu_ps(&C(i, j + 3), _mm_add_ps(c3, _mm_loadu_ps(&C(i, j + 3))));
        }
    }
#endif
}

void GemmSIMD::macro_kernel_8x4_sgemm_intel(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA,
                                            const float *B, int LDB, float beta, float *C, int LDC)
{
#ifndef __APPLE__ // AVX is not available on Apple Silicon

    size_t i, j;

    __m256 valpha = _mm256_set1_ps(alpha);

    // Loop over the 8x4 tiles
    for (i = 0; i + 7 < M; i += 8)
    {
        for (j = 0; j + 3 < N; j += 4)
        {

            // Initialize accumulators for the 4 columns
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            // Perform the actual multiplication for the 8x4 block
            for (size_t k = 0; k < K; ++k)
            {
                // Load a column of A and scale by alpha
                __m256 a = _mm256_mul_ps(valpha, _mm256_loadu_ps(&A(i, k)));

                // Broadcast each scalar from B for each column in the 8x4 block
                __m256 b0 = _mm256_set1_ps(B(k, j + 0));
                __m256 b1 = _mm256_set1_ps(B(k, j + 1));
                __m256 b2 = _mm256_set1_ps(B(k, j + 2));
                __m256 b3 = _mm256_set1_ps(B(k, j + 3));

                // Perform the multiply-accumulate operation
                c0 = _mm256_fmadd_ps(a, b0, c0);
                c1 = _mm256_fmadd_ps(a, b1, c1);
                c2 = _mm256_fmadd_ps(a, b2, c2);
                c3 = _mm256_fmadd_ps(a, b3, c3);
            }

            // Store the final results back to C
            _mm256_storeu_ps(&C(i, j + 0), c0);
            _mm256_storeu_ps(&C(i, j + 1), c1);
            _mm256_storeu_ps(&C(i, j + 2), c2);
            _mm256_storeu_ps(&C(i, j + 3), c3);
        }
    }
#endif
}

// void GemmAVX::macro_kernel_16x4_sgemm_intel(
//     size_t M, size_t N, size_t K,
//     float alpha,
//     const float *A, int LDA,
//     const float *B, int LDB,
//     float beta,
//     float *C, int LDC
// ) {
// #ifndef __APPLE__

//     __m512 valpha = _mm512_set1_ps(alpha);

//     for (size_t j = 0; j + 3 < N; j += 4) {
//         for (size_t i = 0; i + 15 < M; i += 16) {

//             // Accumulators
//             __m512 c0 = _mm512_loadu_ps(&C(i, j + 0));
//             __m512 c1 = _mm512_loadu_ps(&C(i, j + 1));
//             __m512 c2 = _mm512_loadu_ps(&C(i, j + 2));
//             __m512 c3 = _mm512_loadu_ps(&C(i, j + 3));

//             for (size_t k = 0; k < K; ++k) {
//                 __m512 a = _mm512_mul_ps(valpha, _mm512_loadu_ps(&A(i, k)));

//                 __m512 b0 = _mm512_set1_ps(B(k, j + 0));
//                 __m512 b1 = _mm512_set1_ps(B(k, j + 1));
//                 __m512 b2 = _mm512_set1_ps(B(k, j + 2));
//                 __m512 b3 = _mm512_set1_ps(B(k, j + 3));

//                 c0 = _mm512_fmadd_ps(a, b0, c0);
//                 c1 = _mm512_fmadd_ps(a, b1, c1);
//                 c2 = _mm512_fmadd_ps(a, b2, c2);
//                 c3 = _mm512_fmadd_ps(a, b3, c3);
//             }

//             _mm512_storeu_ps(&C(i, j + 0), c0);
//             _mm512_storeu_ps(&C(i, j + 1), c1);
//             _mm512_storeu_ps(&C(i, j + 2), c2);
//             _mm512_storeu_ps(&C(i, j + 3), c3);
//         }
//     }

// #endif
// }

// void GemmAVX::macro_kernel_4x4_sgemm_intel_packed(
//     size_t M, size_t N, size_t K,
//     float alpha,
//     const float *A, int LDA,
//     const float *B, int LDB,
//     float beta,
//     float *C, int LDC
// ) {
// #ifndef __APPLE__
//     __m128 valpha = _mm_set1_ps(alpha);

//     // Allocate temp buffers on stack for packed A and B
//     alignas(16) float packA[4 * K];
//     alignas(16) float packB[4 * K];

//     for (size_t i = 0; i < M; i += 4) {
//         for (size_t j = 0; j < N; j += 4) {

//             // === Pack A (4 x K block) ===
//             for (size_t k = 0; k < 4; ++k) {
//                 packA[k * 4 + 0] = A(i + 0, k);
//                 packA[k * 4 + 1] = A(i + 1, k);
//                 packA[k * 4 + 2] = A(i + 2, k);
//                 packA[k * 4 + 3] = A(i + 3, k);
//             }

//             // === Pack B (4 x 4 block) ===
//             for (size_t k = 0; k < 4; ++k) {
//                 packB[0 * K + k] = B(k, j + 0);
//                 packB[1 * K + k] = B(k, j + 1);
//                 packB[2 * K + k] = B(k, j + 2);
//                 packB[3 * K + k] = B(k, j + 3);
//             }

//             // === Compute C[4x4] block ===
//             __m128 c0 = _mm_setzero_ps();
//             __m128 c1 = _mm_setzero_ps();
//             __m128 c2 = _mm_setzero_ps();
//             __m128 c3 = _mm_setzero_ps();

//             for (size_t k = 0; k < K; ++k) {
//                 __m128 a = _mm_mul_ps(valpha, _mm_load_ps(&packA[k * 4]));

//                 __m128 b0 = _mm_set1_ps(packB[0 * K + k]);
//                 __m128 b1 = _mm_set1_ps(packB[1 * K + k]);
//                 __m128 b2 = _mm_set1_ps(packB[2 * K + k]);
//                 __m128 b3 = _mm_set1_ps(packB[3 * K + k]);

//                 c0 = _mm_add_ps(_mm_mul_ps(a, b0), c0);
//                 c1 = _mm_add_ps(_mm_mul_ps(a, b1), c1);
//                 c2 = _mm_add_ps(_mm_mul_ps(a, b2), c2);
//                 c3 = _mm_add_ps(_mm_mul_ps(a, b3), c3);
//             }

//             // Store result back to C
//             _mm_storeu_ps(&C(i, j + 0), _mm_add_ps(c0, _mm_loadu_ps(&C(i, j + 0))));
//             _mm_storeu_ps(&C(i, j + 1), _mm_add_ps(c1, _mm_loadu_ps(&C(i, j + 1))));
//             _mm_storeu_ps(&C(i, j + 2), _mm_add_ps(c2, _mm_loadu_ps(&C(i, j + 2))));
//             _mm_storeu_ps(&C(i, j + 3), _mm_add_ps(c3, _mm_loadu_ps(&C(i, j + 3))));
//         }
//     }
// #endif
// }

} // namespace gemm

#endif