# GEMM Optimization Project

This project implements and benchmarks various optimized versions of the General Matrix Multiplication (GEMM) operation on both CPUs and GPUs.

## Overview

The General Matrix Multiplication (GEMM) is a fundamental operation in linear algebra: C = α×A×B + β×C, where A, B, and C are matrices and α and β are scalars. This project aims to explore different optimization techniques to maximize GEMM performance, including:

- CPU optimizations with SIMD instructions (AVX, NEON)
- Multi-threading with OpenMP
- GPU acceleration with Metal (on macOS/Apple Silicon)
- Comparison with optimized BLAS implementations

## File Structure

```
Project/
├── Makefile                      # Main GNU makefile
├── README.md                     # Project documentation
├── include/                      # Header files
│   ├── gemm.h                    # GEMM interface definition
│   ├── types.h                   # Matrix class definition
│   ├── benchmark.h               # Benchmarking utilities
│   └── common.h                  # Common utilities
├── src/                          # Source files
│   ├── main.cpp                  # Main application entry
│   ├── benchmark.cpp             # Benchmarking implementation
│   ├── implementations/          # GEMM implementations
│   │   ├── gemm_naive.cpp        # Naive implementation
│   │   ├── gemm_simd.cpp         # SIMD-optimized implementation
│   │   ├── gemm_omp.cpp          # OpenMP implementation
│   │   ├── gemm_blas.cpp         # BLAS wrapper implementation
│   │   └── gemm_metal.cpp        # Metal GPU implementation
│   └── rust_metal_gemm/          # Rust-based Metal implementation
│       ├── Cargo.toml            # Rust project configuration
│       ├── src/                  # Rust source code
│       │   ├── lib.rs            # Rust library implementation
│       │   └── gemm_kernel.metal # Metal shader code
│       └── include/              # C/C++ headers for Rust FFI
│           └── metal_gemm.h      # C interface for Metal implementation
├── build/                        # Build artifacts (created by make)
└── results/                      # Benchmark results (created by the program)
```

## Requirements

### Makefile

### For macOS/Apple Silicon

```shell
# Install required packages
brew install clang-format  # For code formatting
brew install libomp        # OpenMP support
brew install openblas      # Optimized BLAS implementation

# Install Rust (for Metal GPU implementation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

### For Linux/x86_64
```shell
# Install required packages
sudo apt-get install -y clang-format libopenblas-dev libomp-dev
```

Please modify the following variables in `Makefile` to configure your environment. 
```makefile
OMP_PATH := /opt/homebrew/opt/libomp
MKL_PATH := /opt/intel/oneapi/mkl/latest
BLAS_PATH := /opt/homebrew/opt/openblas
```



## Building the Project

To build the project:
```shell
make
```

This will compile the C++ code and, if on Apple Silicon, also build the Rust-based Metal implementation.

## Running Benchmarks

Run with default settings (4 threads):
```shell
make run
```

Run in single-thread mode:
```shell
make run-single
```

Custom thread configuration:
```shell
OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./gemm
```

## Implemented Optimizations

1. **Naive CPU Implementation** - Basic triple-loop matrix multiplication
2. **SIMD CPU Implementation** - Uses architecture-specific SIMD instructions (AVX2/AVX512 on x86, NEON on ARM)
3. **OpenMP CPU Implementation** - Multi-threaded implementation with cache-aware blocking
4. **BLAS Implementation** - Wrapper around optimized BLAS libraries
5. **Metal GPU Implementation** - Uses Apple's Metal compute API for GPU acceleration (Apple Silicon only)

## Performance Tuning

The performance of the GEMM implementations can be tuned via several parameters:

- **Block Sizes**: The blocking parameters in `gemm_simd.cpp` and `gemm_omp.cpp` can be adjusted:
  ```cpp
  #define M_BLOCKING 32   // Block size for M dimension
  #define N_BLOCKING 64   // Block size for N dimension
  #define K_BLOCKING 64   // Block size for K dimension
  ```

- **Thread Count**: The number of OpenMP threads can be set via environment variables:
  ```shell
  OMP_NUM_THREADS=8 ./gemm
  ```

## License


## Acknowledgments

This project is inspired by research in matrix multiplication optimization, including:
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)
- [GEMM: From Pure C to SSE Optimized Micro Kernels](https://github.com/flame/how-to-optimize-gemm)