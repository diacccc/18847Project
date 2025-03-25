#include "gemm.h"
#include <unordered_map>
#include <functional>

namespace gemm {

// Implementation of NaiveCpuGemm::execute
void NaiveCpuGemm::execute(
    float alpha,
    const Matrix<float>& A,
    const Matrix<float>& B,
    float beta,
    Matrix<float>& C
) {
    const size_t M = A.rows();
    const size_t N = B.cols();
    const size_t K = A.cols();
    
    // Check dimensions
    if (B.rows() != K || C.rows() != M || C.cols() != N) {
        std::cerr << "Error: Matrix dimensions don't match for GEMM operation" << std::endl;
        return;
    }
    
    // Apply beta to C
    if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C.at(i, j) *= beta;
            }
        }
    }
    
    // Perform naive matrix multiplication: C = alpha * A * B + C
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