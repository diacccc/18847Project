#include "gemm_blas.h"
#include <unordered_map>
#include <functional>
#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

namespace gemm {

// Implementation of NaiveCpuGemm::execute
void GemmBLAS::execute(
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
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
        M, N, K, 
        alpha, A.data(), A.ld(), B.data(), B.ld(), 
        beta, C.data(), C.ld());

}

} // namespace gemm