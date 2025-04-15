#include "gemm.h"

namespace gemm {
    class GemmSIMD : public GemmImplementation {
        public:
            std::string getName() const override { return "cpu_simd"; }
            
            void execute(
                float alpha,
                const Matrix<float>& A,
                const Matrix<float>& B,
                float beta,
                Matrix<float>& C
            ) override;
            
            #ifdef __APPLE__

            void macro_kernel_4x4_sgemm_neon(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );  

            void packing_A_8_neon(
                const float* A, 
                size_t M, size_t K, size_t LDA,
                float *packed_A
            );

            void packing_A_16_neon(
                const float* A, 
                size_t M, size_t K, size_t LDA,
                float *packed_A
            );

            void packing_B_4_neon(
                const float* B, 
                size_t K, size_t N, size_t LDB,
                float *packed_B
            );

            void packing_B_8_neon(
                const float* B, 
                size_t K, size_t N, size_t LDB,
                float *packed_B
            );
            
            void macro_kernel_8x4_sgemm_neon(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );
            
            void macro_kernel_16x4_sgemm_neon(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );

            void macro_kernel_8x8_sgemm_neon(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );

            void macro_kernel_16x8_sgemm_neon(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );


            #else 

            void macro_kernel_4x1_sgemm_intel(
                size_t M, size_t N, size_t K, 
                float alpha, 
                const float *A, int LDA, 
                const float *B, int LDB, 
                float beta, 
                float *C, int LDC
            );
            
            #endif
    };
}