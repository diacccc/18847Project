#ifdef __APPLE__
#include <arm_acle.h>
#include <arm_neon.h>
#include <functional>
#include <unordered_map>

#include "gemm_simd.h"

#define A(i, j) A[(i) + (j) * LDA]
#define B(i, j) B[(i) + (j) * LDB]
#define C(i, j) C[(i) + (j) * LDC]

namespace gemm
{

void GemmSIMD::packing_A_8_neon(const float *A, size_t M, size_t K, size_t LDA, float *packed_A)
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

void GemmSIMD::packing_A_16_neon(const float *A, size_t M, size_t K, size_t LDA, float *packed_A)
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

void GemmSIMD::packing_B_4_neon(const float *B, size_t K, size_t N, size_t LDB, float *packed_B)
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

void GemmSIMD::packing_B_8_neon(const float *B, size_t K, size_t N, size_t LDB, float *packed_B)
{
    const float *B_ptr0, *B_ptr1, *B_ptr2, *B_ptr3, *B_ptr4, *B_ptr5, *B_ptr6, *B_ptr7;
    float *dst = packed_B;
    for (size_t j = 0; j < N; j += 8)
    {
        B_ptr0 = &B(0, j);
        B_ptr1 = &B(0, j + 1);
        B_ptr2 = &B(0, j + 2);
        B_ptr3 = &B(0, j + 3);
        B_ptr4 = &B(0, j + 4);
        B_ptr5 = &B(0, j + 5);
        B_ptr6 = &B(0, j + 6);
        B_ptr7 = &B(0, j + 7);
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
            *dst = *B_ptr4++;
            dst++;
            *dst = *B_ptr5++;
            dst++;
            *dst = *B_ptr6++;
            dst++;
            *dst = *B_ptr7++;
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
    b = vld1q_f32(packed_B);                                                                                           \
    c00 = vfmaq_laneq_f32(c00, a0, b, 0);                                                                              \
    c01 = vfmaq_laneq_f32(c01, a1, b, 0);                                                                              \
    c10 = vfmaq_laneq_f32(c10, a0, b, 1);                                                                              \
    c11 = vfmaq_laneq_f32(c11, a1, b, 1);                                                                              \
    c20 = vfmaq_laneq_f32(c20, a0, b, 2);                                                                              \
    c21 = vfmaq_laneq_f32(c21, a1, b, 2);                                                                              \
    c30 = vfmaq_laneq_f32(c30, a0, b, 3);                                                                              \
    c31 = vfmaq_laneq_f32(c31, a1, b, 3);                                                                              \
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
            float32x4_t b;
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

#define KERNEL_16x4_SGEMM_NEON                                                                                         \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A));                                                                       \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4));                                                                   \
    a2 = vmulq_f32(valpha, vld1q_f32(packed_A + 8));                                                                   \
    a3 = vmulq_f32(valpha, vld1q_f32(packed_A + 12));                                                                  \
    b = vld1q_f32(packed_B);                                                                                           \
    c00 = vfmaq_laneq_f32(c00, a0, b, 0);                                                                              \
    c01 = vfmaq_laneq_f32(c01, a1, b, 0);                                                                              \
    c02 = vfmaq_laneq_f32(c02, a2, b, 0);                                                                              \
    c03 = vfmaq_laneq_f32(c03, a3, b, 0);                                                                              \
    c10 = vfmaq_laneq_f32(c10, a0, b, 1);                                                                              \
    c11 = vfmaq_laneq_f32(c11, a1, b, 1);                                                                              \
    c12 = vfmaq_laneq_f32(c12, a2, b, 1);                                                                              \
    c13 = vfmaq_laneq_f32(c13, a3, b, 1);                                                                              \
    c20 = vfmaq_laneq_f32(c20, a0, b, 2);                                                                              \
    c21 = vfmaq_laneq_f32(c21, a1, b, 2);                                                                              \
    c22 = vfmaq_laneq_f32(c22, a2, b, 2);                                                                              \
    c23 = vfmaq_laneq_f32(c23, a3, b, 2);                                                                              \
    c30 = vfmaq_laneq_f32(c30, a0, b, 3);                                                                              \
    c31 = vfmaq_laneq_f32(c31, a1, b, 3);                                                                              \
    c32 = vfmaq_laneq_f32(c32, a2, b, 3);                                                                              \
    c33 = vfmaq_laneq_f32(c33, a3, b, 3);                                                                              \
    packed_A += 16;                                                                                                    \
    packed_B += 4;                                                                                                     \
    k++;

void GemmSIMD::macro_kernel_16x4_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int,
                                            const float *B, int, float, float *C, int LDC)
{
    const float *packed_A = A;
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 16)
    {
        for (size_t j = 0; j < N; j += 4)
        {
            float32x4_t a0, a1, a2, a3;
            float32x4_t b;
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

#define KERNEL_8x8_SGEMM_NEON                                                                                          \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A));                                                                       \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4));                                                                   \
    b0 = vld1q_f32(packed_B);                                                                                          \
    b1 = vld1q_f32(packed_B + 4);                                                                                      \
    c00 = vfmaq_laneq_f32(c00, a0, b0, 0);                                                                             \
    c01 = vfmaq_laneq_f32(c01, a1, b0, 0);                                                                             \
    c10 = vfmaq_laneq_f32(c10, a0, b0, 1);                                                                             \
    c11 = vfmaq_laneq_f32(c11, a1, b0, 1);                                                                             \
    c20 = vfmaq_laneq_f32(c20, a0, b0, 2);                                                                             \
    c21 = vfmaq_laneq_f32(c21, a1, b0, 2);                                                                             \
    c30 = vfmaq_laneq_f32(c30, a0, b0, 3);                                                                             \
    c31 = vfmaq_laneq_f32(c31, a1, b0, 3);                                                                             \
    c40 = vfmaq_laneq_f32(c40, a0, b1, 0);                                                                             \
    c41 = vfmaq_laneq_f32(c41, a1, b1, 0);                                                                             \
    c50 = vfmaq_laneq_f32(c50, a0, b1, 1);                                                                             \
    c51 = vfmaq_laneq_f32(c51, a1, b1, 1);                                                                             \
    c60 = vfmaq_laneq_f32(c60, a0, b1, 2);                                                                             \
    c61 = vfmaq_laneq_f32(c61, a1, b1, 2);                                                                             \
    c70 = vfmaq_laneq_f32(c70, a0, b1, 3);                                                                             \
    c71 = vfmaq_laneq_f32(c71, a1, b1, 3);                                                                             \
    packed_A += 8;                                                                                                     \
    packed_B += 8;                                                                                                     \
    k++;

void GemmSIMD::macro_kernel_8x8_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int,
                                           const float *B, int, float, float *C, int LDC)
{
    const float *packed_A = A;
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 8)
    {
        for (size_t j = 0; j < N; j += 8)
        {
            float32x4_t a0, a1;
            float32x4_t b0, b1;
            float32x4_t c00, c01, c10, c11, c20, c21, c30, c31;
            float32x4_t c40, c41, c50, c51, c60, c61, c70, c71;
            c00 = vdupq_n_f32(0);
            c01 = vdupq_n_f32(0);
            c10 = vdupq_n_f32(0);
            c11 = vdupq_n_f32(0);
            c20 = vdupq_n_f32(0);
            c21 = vdupq_n_f32(0);
            c30 = vdupq_n_f32(0);
            c31 = vdupq_n_f32(0);
            c40 = vdupq_n_f32(0);
            c41 = vdupq_n_f32(0);
            c50 = vdupq_n_f32(0);
            c51 = vdupq_n_f32(0);
            c60 = vdupq_n_f32(0);
            c61 = vdupq_n_f32(0);
            c70 = vdupq_n_f32(0);
            c71 = vdupq_n_f32(0);

            packed_A = A + i * K;
            packed_B = B + j * K;

            for (size_t k = 0; k < K;)
            {
                KERNEL_8x8_SGEMM_NEON KERNEL_8x8_SGEMM_NEON KERNEL_8x8_SGEMM_NEON KERNEL_8x8_SGEMM_NEON
            }
            vst1q_f32(&C(i, j), vaddq_f32(c00, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i + 4, j), vaddq_f32(c01, vld1q_f32(&C(i + 4, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c10, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i + 4, j + 1), vaddq_f32(c11, vld1q_f32(&C(i + 4, j + 1))));
            vst1q_f32(&C(i, j + 2), vaddq_f32(c20, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i + 4, j + 2), vaddq_f32(c21, vld1q_f32(&C(i + 4, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c30, vld1q_f32(&C(i, j + 3))));
            vst1q_f32(&C(i + 4, j + 3), vaddq_f32(c31, vld1q_f32(&C(i + 4, j + 3))));

            vst1q_f32(&C(i, j + 4), vaddq_f32(c40, vld1q_f32(&C(i, j + 4))));
            vst1q_f32(&C(i + 4, j + 4), vaddq_f32(c41, vld1q_f32(&C(i + 4, j + 4))));
            vst1q_f32(&C(i, j + 5), vaddq_f32(c50, vld1q_f32(&C(i, j + 5))));
            vst1q_f32(&C(i + 4, j + 5), vaddq_f32(c51, vld1q_f32(&C(i + 4, j + 5))));
            vst1q_f32(&C(i, j + 6), vaddq_f32(c60, vld1q_f32(&C(i, j + 6))));
            vst1q_f32(&C(i + 4, j + 6), vaddq_f32(c61, vld1q_f32(&C(i + 4, j + 6))));
            vst1q_f32(&C(i, j + 7), vaddq_f32(c70, vld1q_f32(&C(i, j + 7))));
            vst1q_f32(&C(i + 4, j + 7), vaddq_f32(c71, vld1q_f32(&C(i + 4, j + 7))));
        }
    }
}

#define KERNEL_16x8_SGEMM_NEON                                                                                         \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A));                                                                       \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4));                                                                   \
    a2 = vmulq_f32(valpha, vld1q_f32(packed_A + 8));                                                                   \
    a3 = vmulq_f32(valpha, vld1q_f32(packed_A + 12));                                                                  \
    b0 = vld1q_f32(packed_B);                                                                                          \
    b1 = vld1q_f32(packed_B + 4);                                                                                      \
    c00 = vfmaq_laneq_f32(c00, a0, b0, 0);                                                                             \
    c01 = vfmaq_laneq_f32(c01, a1, b0, 0);                                                                             \
    c02 = vfmaq_laneq_f32(c02, a2, b0, 0);                                                                             \
    c03 = vfmaq_laneq_f32(c03, a3, b0, 0);                                                                             \
    c10 = vfmaq_laneq_f32(c10, a0, b0, 1);                                                                             \
    c11 = vfmaq_laneq_f32(c11, a1, b0, 1);                                                                             \
    c12 = vfmaq_laneq_f32(c12, a2, b0, 1);                                                                             \
    c13 = vfmaq_laneq_f32(c13, a3, b0, 1);                                                                             \
    c20 = vfmaq_laneq_f32(c20, a0, b0, 2);                                                                             \
    c21 = vfmaq_laneq_f32(c21, a1, b0, 2);                                                                             \
    c22 = vfmaq_laneq_f32(c22, a2, b0, 2);                                                                             \
    c23 = vfmaq_laneq_f32(c23, a3, b0, 2);                                                                             \
    c30 = vfmaq_laneq_f32(c30, a0, b0, 3);                                                                             \
    c31 = vfmaq_laneq_f32(c31, a1, b0, 3);                                                                             \
    c32 = vfmaq_laneq_f32(c32, a2, b0, 3);                                                                             \
    c33 = vfmaq_laneq_f32(c33, a3, b0, 3);                                                                             \
    c40 = vfmaq_laneq_f32(c40, a0, b1, 0);                                                                             \
    c41 = vfmaq_laneq_f32(c41, a1, b1, 0);                                                                             \
    c42 = vfmaq_laneq_f32(c42, a2, b1, 0);                                                                             \
    c43 = vfmaq_laneq_f32(c43, a3, b1, 0);                                                                             \
    c50 = vfmaq_laneq_f32(c50, a0, b1, 1);                                                                             \
    c51 = vfmaq_laneq_f32(c51, a1, b1, 1);                                                                             \
    c52 = vfmaq_laneq_f32(c52, a2, b1, 1);                                                                             \
    c53 = vfmaq_laneq_f32(c53, a3, b1, 1);                                                                             \
    c60 = vfmaq_laneq_f32(c60, a0, b1, 2);                                                                             \
    c61 = vfmaq_laneq_f32(c61, a1, b1, 2);                                                                             \
    c62 = vfmaq_laneq_f32(c62, a2, b1, 2);                                                                             \
    c63 = vfmaq_laneq_f32(c63, a3, b1, 2);                                                                             \
    c70 = vfmaq_laneq_f32(c70, a0, b1, 3);                                                                             \
    c71 = vfmaq_laneq_f32(c71, a1, b1, 3);                                                                             \
    c72 = vfmaq_laneq_f32(c72, a2, b1, 3);                                                                             \
    c73 = vfmaq_laneq_f32(c73, a3, b1, 3);                                                                             \
    packed_A += 16;                                                                                                    \
    packed_B += 8;                                                                                                     \
    k++;

void GemmSIMD::macro_kernel_16x8_sgemm_neon(size_t M, size_t N, size_t K, float alpha, const float *A, int,
                                            const float *B, int, float, float *C, int LDC)
{
    const float *packed_A = A;
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 16)
    {
        for (size_t j = 0; j < N; j += 8)
        {
            float32x4_t a0, a1, a2, a3;
            float32x4_t b0, b1;
            float32x4_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, c40, c41, c42,
                c43, c50, c51, c52, c53, c60, c61, c62, c63, c70, c71, c72, c73;
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

            c40 = vdupq_n_f32(0);
            c41 = vdupq_n_f32(0);
            c42 = vdupq_n_f32(0);
            c43 = vdupq_n_f32(0);

            c50 = vdupq_n_f32(0);
            c51 = vdupq_n_f32(0);
            c52 = vdupq_n_f32(0);
            c53 = vdupq_n_f32(0);

            c60 = vdupq_n_f32(0);
            c61 = vdupq_n_f32(0);
            c62 = vdupq_n_f32(0);
            c63 = vdupq_n_f32(0);

            c70 = vdupq_n_f32(0);
            c71 = vdupq_n_f32(0);
            c72 = vdupq_n_f32(0);
            c73 = vdupq_n_f32(0);

            packed_A = A + i * K;
            packed_B = B + j * K;

            for (size_t k = 0; k < K;)
            {
                KERNEL_16x8_SGEMM_NEON KERNEL_16x8_SGEMM_NEON KERNEL_16x8_SGEMM_NEON KERNEL_16x8_SGEMM_NEON
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

            vst1q_f32(&C(i, j + 4), vaddq_f32(c40, vld1q_f32(&C(i, j + 4))));
            vst1q_f32(&C(i + 4, j + 4), vaddq_f32(c41, vld1q_f32(&C(i + 4, j + 4))));
            vst1q_f32(&C(i + 8, j + 4), vaddq_f32(c42, vld1q_f32(&C(i + 8, j + 4))));
            vst1q_f32(&C(i + 12, j + 4), vaddq_f32(c43, vld1q_f32(&C(i + 12, j + 4))));

            vst1q_f32(&C(i, j + 5), vaddq_f32(c50, vld1q_f32(&C(i, j + 5))));
            vst1q_f32(&C(i + 4, j + 5), vaddq_f32(c51, vld1q_f32(&C(i + 4, j + 5))));
            vst1q_f32(&C(i + 8, j + 5), vaddq_f32(c52, vld1q_f32(&C(i + 8, j + 5))));
            vst1q_f32(&C(i + 12, j + 5), vaddq_f32(c53, vld1q_f32(&C(i + 12, j + 5))));

            vst1q_f32(&C(i, j + 6), vaddq_f32(c60, vld1q_f32(&C(i, j + 6))));
            vst1q_f32(&C(i + 4, j + 6), vaddq_f32(c61, vld1q_f32(&C(i + 4, j + 6))));
            vst1q_f32(&C(i + 8, j + 6), vaddq_f32(c62, vld1q_f32(&C(i + 8, j + 6))));
            vst1q_f32(&C(i + 12, j + 6), vaddq_f32(c63, vld1q_f32(&C(i + 12, j + 6))));

            vst1q_f32(&C(i, j + 7), vaddq_f32(c70, vld1q_f32(&C(i, j + 7))));
            vst1q_f32(&C(i + 4, j + 7), vaddq_f32(c71, vld1q_f32(&C(i + 4, j + 7))));
            vst1q_f32(&C(i + 8, j + 7), vaddq_f32(c72, vld1q_f32(&C(i + 8, j + 7))));
            vst1q_f32(&C(i + 12, j + 7), vaddq_f32(c73, vld1q_f32(&C(i + 12, j + 7))));
        }
    }
}

} // namespace gemm

#endif