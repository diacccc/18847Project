#include "gemm_avx.h"
#include <unordered_map>
#include <functional>
#ifdef __APPLE__
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #include <simde/x86/avx2.h>
#else
  #include <immintrin.h>
#endif
namespace gemm {

// Implementation of NaiveCpuGemm::execute
void GemmAVX::execute(
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
    
    __m128 valpha = _mm_set1_ps(alpha);
    for (size_t i = 0; i < M; i += 4) {
        for (size_t j = 0; j < N; j += 4) {
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                __m128 a = _mm_mul_ps(valpha, _mm_loadu_ps(&A.at(i, k)));
                __m128 b0 = _mm_set1_ps(B.at(k, j));
                __m128 b1 = _mm_set1_ps(B.at(k, j + 1));
                __m128 b2 = _mm_set1_ps(B.at(k, j + 2));
                __m128 b3 = _mm_set1_ps(B.at(k, j + 3));
                
                // Apple Silicon has no fma instruction
                // c0 = _mm_fmadd_ps(a, b0, c0);
                // c1 = _mm_fmadd_ps(a, b1, c1);
                // c2 = _mm_fmadd_ps(a, b2, c2);
                // c3 = _mm_fmadd_ps(a, b3, c3);
                c0 = _mm_add_ps(_mm_mul_ps(a, b0), c0);
                c1 = _mm_add_ps(_mm_mul_ps(a, b1), c1);
                c2 = _mm_add_ps(_mm_mul_ps(a, b2), c2);
                c3 = _mm_add_ps(_mm_mul_ps(a, b3), c3);
            }
            _mm_storeu_ps(&C.at(i, j    ), _mm_add_ps(c0, _mm_loadu_ps(&C.at(i, j   ))));
            _mm_storeu_ps(&C.at(i, j + 1), _mm_add_ps(c1, _mm_loadu_ps(&C.at(i, j + 1))));
            _mm_storeu_ps(&C.at(i, j + 2), _mm_add_ps(c2, _mm_loadu_ps(&C.at(i, j + 2))));
            _mm_storeu_ps(&C.at(i, j + 3), _mm_add_ps(c3, _mm_loadu_ps(&C.at(i, j + 3))));
        }
    }
}

}