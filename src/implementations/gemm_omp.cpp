/**
* @file OMP implementation of GEMM
* @author Molly Xiao minxiao@andrew.cmu.edu
* @created 2025-04-12
*/
#include "gemm_omp.h"
#include <omp.h>
#include <iostream>
#include <cassert>

namespace gemm {
    void GemmOMP::execute(
        float alpha,
        const Matrix<float>& A,
        const Matrix<float>& B,
        float beta,
        Matrix<float>& C
    ) {
        const size_t M = A.rows();
        const size_t N = B.cols();
        const size_t K = A.cols();

        assert(B.rows() == K);
        assert(C.rows() == M);
        assert(C.cols() == N);

        scale(beta, C);

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) += alpha * sum;
            }
        }
    }
} // namespace gemm
