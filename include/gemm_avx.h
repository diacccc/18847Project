#include "gemm.h"
namespace gemm {
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

            void kernel_4x1_sgemm(
                    size_t M, size_t N, size_t K, 
                    float alpha, 
                    float *A, int LDA, 
                    float *B, int LDB, 
                    float beta, 
                    float *C, int LDC
            );
        };
}