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
    };

}

#endif //GEMM_OMP_H

