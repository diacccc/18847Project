#pragma once

#include <string>
#include "types.h"

namespace gemm {

// GEMM Operation: C = alpha * A * B + beta * C
// Where A is (M x K), B is (K x N), and C is (M x N)
    class GemmImplementation {
    public:
        virtual ~GemmImplementation() = default;

        // Return the name of this implementation
        virtual std::string getName() const = 0;
        
        // Execute the GEMM operation
        virtual void execute(
            float alpha,                 // Scalar alpha
            const Matrix<float>& A,      // Matrix A (M x K)
            const Matrix<float>& B,      // Matrix B (K x N)
            float beta,                  // Scalar beta
            Matrix<float>& C             // Matrix C (M x N)
        ) = 0;
        
        // Optional: Initialize any resources required by this implementation
        virtual bool initialize() { return true; }
        
        // Optional: Clean up any resources used by this implementation
        virtual void cleanup() {}

        inline void scale(float beta, Matrix<float>& C) {
            if (beta != 1.0f) {
                for (size_t i = 0; i < C.rows(); ++i) {
                    for (size_t j = 0; j < C.cols(); ++j) {
                        C.at(i, j) *= beta;
                    }
                }
            }
        }
    };
} // namespace gemm