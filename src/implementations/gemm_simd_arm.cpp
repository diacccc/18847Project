#ifdef __APPLE__
#include "gemm_avx.h"
#include <unordered_map>
#include <functional>
#include <arm_neon.h>

#define A(i, j) A[(i)+(j)*LDA]
#define B(i, j) B[(i)+(j)*LDB]
#define C(i, j) C[(i)+(j)*LDC]

namespace gemm {

void GemmAVX::packing_A_8x4_neon(
    const float* A, 
    size_t M, size_t K, size_t LDA,
    float *packed_A
) {
    float* dst = packed_A;
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < K; ++j) {
            vst1q_f32(dst, vld1q_f32(&A(i, j)));
            vst1q_f32(dst + 4, vld1q_f32(&A(i + 4, j)));
            dst += 8;
        }
    }
}

void GemmAVX::packing_B_8x4_neon(
    const float* B, 
    size_t K, size_t N, size_t LDB,
    float *packed_B
) {
    const float *B_ptr0, *B_ptr1, *B_ptr2, *B_ptr3;
    float *dst = packed_B;
    for (size_t j = 0; j < N; j += 4) {
        B_ptr0 = &B(0, j);
        B_ptr1 = &B(0, j + 1);
        B_ptr2 = &B(0, j + 2);
        B_ptr3 = &B(0, j + 3);
        for (size_t k = 0; k < K; ++k) {
            *dst = *B_ptr0++; dst++;
            *dst = *B_ptr1++; dst++;
            *dst = *B_ptr2++; dst++;
            *dst = *B_ptr3++; dst++;
        }
    }
}

void GemmAVX::macro_kernel_4x4_sgemm_neon(
    size_t M, size_t N, size_t K, 
    float alpha, 
    const float *A, int LDA, 
    const float *B, int LDB, 
    float beta, 
    float *C, int LDC
) {
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
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
            
            for (size_t k = 0; k < K; ++k) {
                float32x4_t a = vmulq_f32(valpha, vld1q_f32(&A(i, k)));
                float32x4_t b0 = vld1q_dup_f32(b_ptr0++);
                float32x4_t b1 = vld1q_dup_f32(b_ptr1++);
                float32x4_t b2 = vld1q_dup_f32(b_ptr2++);
                float32x4_t b3 = vld1q_dup_f32(b_ptr3++);
                
                // On Apple Silicon (ARM), we can use FMA
                #ifdef __ARM_FEATURE_FMA
                c0 = vfmaq_f32(c0, a, b0);
                c1 = vfmaq_f32(c1, a, b1);
                c2 = vfmaq_f32(c2, a, b2);
                c3 = vfmaq_f32(c3, a, b3);
                #else
                c0 = vaddq_f32(vmulq_f32(a, b0), c0);
                c1 = vaddq_f32(vmulq_f32(a, b1), c1);
                c2 = vaddq_f32(vmulq_f32(a, b2), c2);
                c3 = vaddq_f32(vmulq_f32(a, b3), c3);
                #endif
            }
            vst1q_f32(&C(i, j),     vaddq_f32(c0, vld1q_f32(&C(i, j))));
            vst1q_f32(&C(i, j + 1), vaddq_f32(c1, vld1q_f32(&C(i, j + 1))));
            vst1q_f32(&C(i, j + 2), vaddq_f32(c2, vld1q_f32(&C(i, j + 2))));
            vst1q_f32(&C(i, j + 3), vaddq_f32(c3, vld1q_f32(&C(i, j + 3))));
        }
    }
}

#define KERNEL_8x4_SGEMM_NEON \
    a0 = vmulq_f32(valpha, vld1q_f32(packed_A)); \
    a1 = vmulq_f32(valpha, vld1q_f32(packed_A + 4)); \
    b0 = vld1q_dup_f32(packed_B); \
    b1 = vld1q_dup_f32(packed_B + 1); \
    b2 = vld1q_dup_f32(packed_B + 2); \
    b3 = vld1q_dup_f32(packed_B + 3); \
    c00 = vfmaq_f32(c00, a0, b0); \
    c01 = vfmaq_f32(c01, a1, b0); \
    c10 = vfmaq_f32(c10, a0, b1); \
    c11 = vfmaq_f32(c11, a1, b1); \
    c20 = vfmaq_f32(c20, a0, b2); \
    c21 = vfmaq_f32(c21, a1, b2); \
    c30 = vfmaq_f32(c30, a0, b3); \
    c31 = vfmaq_f32(c31, a1, b3); \
    packed_A += 8; packed_B += 4; k++;


void GemmAVX::macro_kernel_8x4_sgemm_neon(
    size_t M, size_t N, size_t K, 
    float alpha, 
    const float *A, int LDA, 
    const float *B, int LDB, 
    float beta, 
    float *C, int LDC
) {
    const float *packed_A = A; 
    const float *packed_B = B;

    float32x4_t valpha = vdupq_n_f32(alpha);
    for (size_t i = 0; i < M; i += 8) {
        for (size_t j = 0; j < N; j += 4) {
            const float *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
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
            
            for (size_t k = 0; k < K;) {
                KERNEL_8x4_SGEMM_NEON
                KERNEL_8x4_SGEMM_NEON
                KERNEL_8x4_SGEMM_NEON
                KERNEL_8x4_SGEMM_NEON
            }
            vst1q_f32(&C(i    , j    ), vaddq_f32(c00, vld1q_f32(&C(i    , j    ))));
            vst1q_f32(&C(i + 4, j    ), vaddq_f32(c01, vld1q_f32(&C(i + 4, j    ))));
            vst1q_f32(&C(i    , j + 1), vaddq_f32(c10, vld1q_f32(&C(i    , j + 1))));
            vst1q_f32(&C(i + 4, j + 1), vaddq_f32(c11, vld1q_f32(&C(i + 4, j + 1))));
            vst1q_f32(&C(i    , j + 2), vaddq_f32(c20, vld1q_f32(&C(i    , j + 2))));
            vst1q_f32(&C(i + 4, j + 2), vaddq_f32(c21, vld1q_f32(&C(i + 4, j + 2))));
            vst1q_f32(&C(i    , j + 3), vaddq_f32(c30, vld1q_f32(&C(i    , j + 3))));
            vst1q_f32(&C(i + 4, j + 3), vaddq_f32(c31, vld1q_f32(&C(i + 4, j + 3))));
        }
    }
}

}

#endif