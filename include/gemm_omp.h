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
        std::string getName() const override { return "cpu_omp"; }

        void execute(
            float alpha,
            const Matrix<float>& A,
            const Matrix<float>& B,
            float beta,
            Matrix<float>& C
        ) override;

	private:
		void pack_block_A(const Matrix<float> &A, float *__restrict__ packed, size_t ib, size_t kb, size_t M, size_t K);
		void pack_block_B(const Matrix<float> &B, float *__restrict__ packed, size_t kb, size_t jb, size_t K, size_t N);
    	void micro_kernel(size_t K, float alpha, const float *__restrict__ A, const float *__restrict__ B,
				  float *__restrict__ C, size_t LDC);
	};
};

#endif //GEMM_OMP_H

