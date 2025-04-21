#pragma once

#include "gemm.h"

namespace gemm {
    // Forward declaration for the Rust Metal GEMM implementation
    struct MetalGEMM;

    class GemmMetal : public GemmImplementation {
    public:
        GemmMetal();
        ~GemmMetal();

        std::string getName() const override { return "Metal"; }
        
        void execute(
            float alpha,
            const Matrix<float>& A,
            const Matrix<float>& B,
            float beta,
            Matrix<float>& C
        ) override;

        bool initialize() override;
        void cleanup() override;

    private:
        void* metal_gemm_impl;  // Opaque pointer to the Rust MetalGEMM
    };
}