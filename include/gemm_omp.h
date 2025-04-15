/**
* @file gemm_omp.h
* @author Molly Xiao minxiao@andrew.cmu.edu
* @created 2025-04-12
*/
#ifndef GEMM_OMP_H
#define GEMM_OMP_H
#include "gemm.h"

namespace gemm {
    class GemmOMP : public GemmImplementation {
    public:
        std::string getName() const override { return "OMP"; }

        void execute(
            float alpha,
            const Matrix<float>& A,
            const Matrix<float>& B,
            float beta,
            Matrix<float>& C
        ) override;

        void macro_kernel_4x4_sgemm(
            size_t M, size_t N, size_t K,
            float alpha,
            const float *A, int LDA,
            const float *B, int LDB,
            float beta,
            float *C, int LDC
        );

        void macro_kernel_8x8_sgemm(
            size_t M, size_t N, size_t K,
            float alpha,
            const float *A, int LDA,
            const float *B, int LDB,
            float beta,
            float *C, int LDC
        );
    };

}

#endif //GEMM_OMP_H

