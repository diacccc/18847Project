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

            void macro_kernel_4x1_sgemm_neon(
                    size_t M, size_t N, size_t K, 
                    float alpha, 
                    const float *A, int LDA, 
                    const float *B, int LDB, 
                    float beta, 
                    float *C, int LDC
            );

            void macro_kernel_4x1_sgemm_intel(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
        );
        };
}