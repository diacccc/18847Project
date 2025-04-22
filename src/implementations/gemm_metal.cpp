#include "gemm_metal.h"
#include <functional>
#include <iostream>
#include <cmath>

// FFI interface to Rust functions
extern "C" {
    void* metal_gemm_create();
    void metal_gemm_execute(
        void* gemm,
        float alpha,
        const float* a,
        size_t a_rows,
        size_t a_cols,
        size_t a_ld,
        const float* b,
        size_t b_rows,
        size_t b_cols,
        size_t b_ld,
        float beta,
        float* c,
        size_t c_rows,
        size_t c_cols,
        size_t c_ld
    );
    void metal_gemm_destroy(void* gemm);
}

namespace gemm {

GemmMetal::GemmMetal() : metal_gemm_impl(nullptr) {
}

GemmMetal::~GemmMetal() {
    cleanup();
}

bool GemmMetal::initialize() {
    if (metal_gemm_impl == nullptr) {
        try {
            metal_gemm_impl = metal_gemm_create();
            if (metal_gemm_impl == nullptr) {
                std::cerr << "Failed to create Metal GEMM implementation" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception during Metal GEMM initialization: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "Unknown exception during Metal GEMM initialization" << std::endl;
            return false;
        }
    }
    
    return true;
}

void GemmMetal::cleanup() {
    if (metal_gemm_impl != nullptr) {
        metal_gemm_destroy(metal_gemm_impl);
        metal_gemm_impl = nullptr;
    }
}

void GemmMetal::execute(
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

    if (metal_gemm_impl == nullptr && !initialize()) {
        std::cerr << "Metal GEMM not initialized, falling back to CPU implementation" << std::endl;
        
        // Fall back to a simple implementation if Metal initialization failed
        scale(beta, C);
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) += alpha * sum;
            }
        }
        
        return;
    }

    // Pre-scale C by beta before passing to Metal
    // This ensures we match the CPU implementation's behavior
    if (beta != 1.0f) {
        scale(beta, C);
        // Use beta=0 for Metal since we've already applied it
        beta = 0.0f;
    }

    try {
        metal_gemm_execute(
            metal_gemm_impl,
            alpha,
            A.data(),
            A.rows(),
            A.cols(),
            A.ld(),
            B.data(),
            B.rows(),
            B.cols(),
            B.ld(),
            beta,
            C.data(),
            C.rows(),
            C.cols(),
            C.ld()
        );
    } catch (const std::exception& e) {
        std::cerr << "Exception during Metal GEMM execution: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown exception during Metal GEMM execution" << std::endl;
        throw;
    }
}

} // namespace gemm