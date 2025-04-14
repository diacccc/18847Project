#include "gemm_naive.h"

#include <functional>
#include <unordered_map>

namespace gemm
{

// Implementation of NaiveCpuGemm::execute
void GemmNaive::execute(float alpha, const Matrix<float> &A, const Matrix<float> &B, float beta, Matrix<float> &C)
{
    const size_t M = A.rows();
    const size_t N = B.cols();
    const size_t K = A.cols();

    assert(B.rows() == K);
    assert(C.rows() == M);
    assert(C.cols() == N);

    scale(beta, C);

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) += alpha * sum;
        }
    }
}

} // namespace gemm