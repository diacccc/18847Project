#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant int& ldc [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int i = gid.x;
    const int j = gid.y;
    
    // Bounds check
    if (i >= M || j >= N) return;
    
    // Use higher precision for accumulation
    float acc = 0.0f;
    
    // Calculate dot product
    for (int k = 0; k < K; k++) {
        // Column-major indexing: i + k * lda and k + j * ldb
        acc += A[i + k * lda] * B[k + j * ldb];
    }
    
    // Apply alpha scaling
    acc *= alpha;
    
    // Apply beta scaling and store result
    if (beta != 0.0f) {
        C[i + j * ldc] = acc + beta * C[i + j * ldc];
    } else {
        C[i + j * ldc] = acc;
    }
}