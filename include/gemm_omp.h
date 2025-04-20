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

	private:
    	// Helper methods
    	void micro_kernel_8x8(size_t K, float alpha, const float *A, const float *B, float *C, size_t LDC);
    	void micro_kernel_4x4(size_t M, size_t N, size_t K, float alpha, const float *A, const float *B, float *C, size_t LDC);
    };
}

#endif //GEMM_OMP_H

