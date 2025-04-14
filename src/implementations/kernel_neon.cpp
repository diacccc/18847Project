#ifdef __APPLE__
#include <arm_neon.h>

#include <functional>
#include <unordered_map>

#include "gemm_simd.h"

#define A(i, j) A[(i) + (j) * LDA]
#define B(i, j) B[(i) + (j) * LDB]
#define C(i, j) C[(i) + (j) * LDC]

namespace gemm
{

void GemmSIMD::packing_A_8x4_neon(const float *A, size_t M, size_t K, size_t LDA, float *packed_A)
{
    float *dst = packed_A;
    for (size_t i = 0; i < M; i += 8)
    {
        for (size_t j = 0; j < K; ++j)
        {
            vst1q_f32(dst, vld1q_f32(&A(i, j)));
            vst1q_f32(dst + 4, vld1q_f32(&A(i + 4, j)));
            dst += 8;
        }
    }
}

void GemmSIMD::packing_B_8x4_neon(const float *B, size_t K, size_t N, size_t LDB, float *packed_B)
{
    const float *B_ptr0, *B_ptr1, *B_ptr2, *B_ptr3;
    float *dst = packed_B;
    for (size_t j = 0; j < N; j += 4)
    {
        B_ptr0 = &B(0, j);
        B_ptr1 = &B(0, j + 1);
        B_ptr2 = &B(0, j + 2);
        B_ptr3 = &B(0, j + 3);
        for (size_t k = 0; k < K; ++k)
        {
            *dst = *B_ptr0++;
            dst++;
            *dst = *B_ptr1++;
            dst++;
            *dst = *B_ptr2++;
            dst++;
            *dst = *B_ptr3++;
            dst++;
        }
    }
}

void GemmSIMD::macro_kernel_4x4_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA,
                                           const float *B, int LDB, float, float *C, int LDC)
{
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 4)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            float32x4_t c0, c1, c2, c3;
            const float *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
            b_ptr0 = &B(0, j);
            b_ptr1 = &B(0, j + 1);
            b_ptr2 = &B(0, j + 2);
            b_ptr3 = &B(0, j + 3);
            c0 = vdupq_n_f32(0);
            c1 = vdupq_n_f32(0);
            c2 = vdupq_n_f32(0);
            c3 = vdupq_n_f32(0);

            for (size_t k = 0; k < K; ++k)
            {
                float32x4_t a = vmulq_f32(valpha, vld1q_f32(&A(i, k)));
                float32x4_t b0 = vld1q_dup_f32(b_ptr0++);
                float32x4_t b1 = vld1q_dup_f32(b_ptr1++);
                float32x4_t b2 = vld1q_dup_f32(b_ptr2++);
                float32x4_t b3 = vld1q_dup_f32(b_ptr3++);

                c0 = vfmaq_f32(c0, a, b0);
                c1 = vfmaq_f32(c1, a, b1);
                c2 = vfmaq_f32(c2, a, b2);
                c3 = vfmaq_f32(c3, a, b3);
            }
            vst1q_f32(&C(i, j), vaddq_f32(c0, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c1, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i, j + 2), vaddq_f32(c2, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c3, vld1q_f32(&C(i, j + 3))));
        }
    }
}

#define KERNEL_8x4_SGEMM_NEON                                                                                          \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A));                                                                       \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4));                                                                   \
    b0 = vld1q_dup_f32(packed_B);                                                                                      \
    b1 = vld1q_dup_f32(packed_B + 1);                                                                                  \
    b2 = vld1q_dup_f32(packed_B + 2);                                                                                  \
    b3 = vld1q_dup_f32(packed_B + 3);                                                                                  \
    c00 = vfmaq_f32(c00, a0, b0);                                                                                      \
    c01 = vfmaq_f32(c01, a1, b0);                                                                                      \
    c10 = vfmaq_f32(c10, a0, b1);                                                                                      \
    c11 = vfmaq_f32(c11, a1, b1);                                                                                      \
    c20 = vfmaq_f32(c20, a0, b2);                                                                                      \
    c21 = vfmaq_f32(c21, a1, b2);                                                                                      \
    c30 = vfmaq_f32(c30, a0, b3);                                                                                      \
    c31 = vfmaq_f32(c31, a1, b3);                                                                                      \
    packed_A += 8;                                                                                                     \
    packed_B += 4;                                                                                                     \
    k++;

void GemmSIMD::macro_kernel_8x4_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int,
                                           const float *B, int, float, float *C, int LDC)
{
    const float *packed_A = A;
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 8)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            float32x4_t a0, a1;
            float32x4_t b0, b1, b2, b3;
            float32x4_t c00, c01, c10, c11, c20, c21, c30, c31;

            c00 = vdupq_n_f32(0);
            c01 = vdupq_n_f32(0);
            c10 = vdupq_n_f32(0);
            c11 = vdupq_n_f32(0);
            c20 = vdupq_n_f32(0);
            c21 = vdupq_n_f32(0);
            c30 = vdupq_n_f32(0);
            c31 = vdupq_n_f32(0);

            packed_A = A + i * K;
            packed_B = B + j * K;

            for (size_t k = 0; k < K;)
            {
                KERNEL_8x4_SGEMM_NEON KERNEL_8x4_SGEMM_NEON KERNEL_8x4_SGEMM_NEON KERNEL_8x4_SGEMM_NEON
            }
            vst1q_f32(&C(i, j), vaddq_f32(c00, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i + 4, j), vaddq_f32(c01, vld1q_f32(&C(i + 4, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c10, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i + 4, j + 1), vaddq_f32(c11, vld1q_f32(&C(i + 4, j + 1))));
            vst1q_f32(&C(i, j + 2), vaddq_f32(c20, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i + 4, j + 2), vaddq_f32(c21, vld1q_f32(&C(i + 4, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c30, vld1q_f32(&C(i, j + 3))));
            vst1q_f32(&C(i + 4, j + 3), vaddq_f32(c31, vld1q_f32(&C(i + 4, j + 3))));
        }
    }
}

void GemmSIMD::packing_A_16x4_neon(const float *A, size_t M, size_t K, size_t LDA, float *packed_A)
{
    float *dst = packed_A;
    for (size_t i = 0; i < M; i += 16)
    {
        for (size_t j = 0; j < K; ++j)
        {
            vst1q_f32(dst, vld1q_f32(&A(i, j)));
            vst1q_f32(dst + 4, vld1q_f32(&A(i + 4, j)));
            vst1q_f32(dst + 8, vld1q_f32(&A(i + 8, j)));
            vst1q_f32(dst + 12, vld1q_f32(&A(i + 12, j)));
            dst += 16;
        }
    }
}

void GemmSIMD::packing_B_16x4_neon(const float *B, size_t K, size_t N, size_t LDB, float *packed_B)
{
    const float *B_ptr0, *B_ptr1, *B_ptr2, *B_ptr3;
    float *dst = packed_B;
    for (size_t j = 0; j < N; j += 4)
    {
        B_ptr0 = &B(0, j);
        B_ptr1 = &B(0, j + 1);
        B_ptr2 = &B(0, j + 2);
        B_ptr3 = &B(0, j + 3);
        for (size_t k = 0; k < K; ++k)
        {
            *dst = *B_ptr0++;
            dst++;
            *dst = *B_ptr1++;
            dst++;
            *dst = *B_ptr2++;
            dst++;
            *dst = *B_ptr3++;
            dst++;
        }
    }
}

#define KERNEL_16x4_SGEMM_NEON                                                                                         \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A));                                                                       \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4));                                                                   \
    a2 = vmulq_f32(valpha, vld1q_f32(packed_A + 8));                                                                   \
    a3 = vmulq_f32(valpha, vld1q_f32(packed_A + 12));                                                                  \
    b0 = vld1q_dup_f32(packed_B);                                                                                      \
    b1 = vld1q_dup_f32(packed_B + 1);                                                                                  \
    b2 = vld1q_dup_f32(packed_B + 2);                                                                                  \
    b3 = vld1q_dup_f32(packed_B + 3);                                                                                  \
    c00 = vfmaq_f32(c00, a0, b0);                                                                                      \
    c01 = vfmaq_f32(c01, a1, b0);                                                                                      \
    c02 = vfmaq_f32(c02, a2, b0);                                                                                      \
    c03 = vfmaq_f32(c03, a3, b0);                                                                                      \
    c10 = vfmaq_f32(c10, a0, b1);                                                                                      \
    c11 = vfmaq_f32(c11, a1, b1);                                                                                      \
    c12 = vfmaq_f32(c12, a2, b1);                                                                                      \
    c13 = vfmaq_f32(c13, a3, b1);                                                                                      \
    c20 = vfmaq_f32(c20, a0, b2);                                                                                      \
    c21 = vfmaq_f32(c21, a1, b2);                                                                                      \
    c22 = vfmaq_f32(c22, a2, b2);                                                                                      \
    c23 = vfmaq_f32(c23, a3, b2);                                                                                      \
    c30 = vfmaq_f32(c30, a0, b3);                                                                                      \
    c31 = vfmaq_f32(c31, a1, b3);                                                                                      \
    c32 = vfmaq_f32(c32, a2, b3);                                                                                      \
    c33 = vfmaq_f32(c33, a3, b3);                                                                                      \
    packed_A += 16;                                                                                                    \
    packed_B += 4;                                                                                                     \
    k++;

void GemmSIMD::macro_kernel_16x4_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int LDA,
                                            const float *B, int LDB, float beta, float *C, int LDC)
{
    const float *packed_A = A;
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 16)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            float32x4_t a0, a1, a2, a3;
            float32x4_t b0, b1, b2, b3;
            float32x4_t c00, c01, c02, c03;
            float32x4_t c10, c11, c12, c13;
            float32x4_t c20, c21, c22, c23;
            float32x4_t c30, c31, c32, c33;

            packed_A = A + i * K;
            packed_B = B + j * K;

            c00 = vdupq_n_f32(0);
            c01 = vdupq_n_f32(0);
            c02 = vdupq_n_f32(0);
            c03 = vdupq_n_f32(0);
            c10 = vdupq_n_f32(0);
            c11 = vdupq_n_f32(0);
            c12 = vdupq_n_f32(0);
            c13 = vdupq_n_f32(0);
            c20 = vdupq_n_f32(0);
            c21 = vdupq_n_f32(0);
            c22 = vdupq_n_f32(0);
            c23 = vdupq_n_f32(0);
            c30 = vdupq_n_f32(0);
            c31 = vdupq_n_f32(0);
            c32 = vdupq_n_f32(0);
            c33 = vdupq_n_f32(0);

            for (size_t k = 0; k < K;)
            {
                KERNEL_16x4_SGEMM_NEON KERNEL_16x4_SGEMM_NEON KERNEL_16x4_SGEMM_NEON KERNEL_16x4_SGEMM_NEON
            }
            vst1q_f32(&C(i, j), vaddq_f32(c00, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i + 4, j), vaddq_f32(c01, vld1q_f32(&C(i + 4, j))));
            vst1q_f32(&C(i + 8, j), vaddq_f32(c02, vld1q_f32(&C(i + 8, j))));
            vst1q_f32(&C(i + 12, j), vaddq_f32(c03, vld1q_f32(&C(i + 12, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c10, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i + 4, j + 1), vaddq_f32(c11, vld1q_f32(&C(i + 4, j + 1))));
            vst1q_f32(&C(i + 8, j + 1), vaddq_f32(c12, vld1q_f32(&C(i + 8, j + 1))));
            vst1q_f32(&C(i + 12, j + 1), vaddq_f32(c13, vld1q_f32(&C(i + 12, j + 1))));

            vst1q_f32(&C(i, j + 2), vaddq_f32(c20, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i + 4, j + 2), vaddq_f32(c21, vld1q_f32(&C(i + 4, j + 2))));
            vst1q_f32(&C(i + 8, j + 2), vaddq_f32(c22, vld1q_f32(&C(i + 8, j + 2))));
            vst1q_f32(&C(i + 12, j + 2), vaddq_f32(c23, vld1q_f32(&C(i + 12, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c30, vld1q_f32(&C(i, j + 3))));
            vst1q_f32(&C(i + 4, j + 3), vaddq_f32(c31, vld1q_f32(&C(i + 4, j + 3))));
            vst1q_f32(&C(i + 8, j + 3), vaddq_f32(c32, vld1q_f32(&C(i + 8, j + 3))));
            vst1q_f32(&C(i + 12, j + 3), vaddq_f32(c33, vld1q_f32(&C(i + 12, j + 3))));
        }
    }
}

} // namespace gemm

#endif