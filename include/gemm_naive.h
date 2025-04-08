#include "gemm.h"

namespace gemm {
    class GemmNaive : public GemmImplementation {
        public:
            std::string getName() const override { return "cpu_naive"; }
            
            void execute(
                float alpha,
                const Matrix<float>& A,
                const Matrix<float>& B,
                float beta,
                Matrix<float>& C
            ) override;
        };
            
}
