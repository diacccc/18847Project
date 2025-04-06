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
};

// Forward declarations of implementation classes
class GemmNaive : public GemmImplementation {
public:
    std::string getName() const override { return "cpu_naive"; }
    
    void execute(
        float alpha,
        const Matrix<float>& A,
        const Matrix<float>& B,
        float beta,
        Matrix<float>& C
    ) override;
};

class GemmAVX : public GemmImplementation {
    public:
        std::string getName() const override { return "cpu_avx"; }
        
        void execute(
            float alpha,
            const Matrix<float>& A,
            const Matrix<float>& B,
            float beta,
            Matrix<float>& C
        ) override;
    };
} // namespace gemm