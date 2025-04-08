#include "gemm.h"

namespace gemm {
    class GemmBLAS : public GemmImplementation {
        public:
            std::string getName() const override { return "BLAS"; }
            
            void execute(
                float alpha,
                const Matrix<float>& A,
                const Matrix<float>& B,
                float beta,
                Matrix<float>& C
            ) override;
        };
            
}
