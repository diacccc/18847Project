# 18847 Project - Optimizing GEMM on both CPUs and GPUs

 ## Synopsis

This project aims to optimize the performance of General Matrix Multiplication (GEMM) by implementing SIMD (Single Instruction, Multiple Data) and multithreading on CPUs, as well as utilizing CUDA/METAL on GPUs. The main objectives are to enhance computational efficiency, reduce execution time, and achieve near-optimal performance. The methodology includes leveraging SIMD instructions and multithreading techniques on CPUs, and implementing CUDA/METAL kernels on GPUs. Expected outcomes include significant performance improvements in GEMM operations, validated through benchmarking and performance analysis by comparing the result with existing BLAS and Intel MKL benchmark. This project is to address critical performance bottlenecks in numerical computing and scientific applications, potentially benefiting a wide range of computational tasks。

## Deliverables

| Feature         | Details                                                      | Resources                                                    | Estimated Time | Assignee |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | -------- |
| Generic Library | Templated C++ GEMM implementation<br />**1. double / float**<br />2. Low-level float point f8e4m3 f8e5m2<br />3. Unit tests |                                                              | 4 hours        |          |
| CPU             | **1. AVX2<br />2. AVX512<br />3. OpenMP**<br />4.TBB         | https://github.com/flame/how-to-optimize-gemm<br />https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf<br />https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html | -              |          |
| GPU             | **1. CUDA<br />2. Metal (MacOS)**<br />3. OpenCL<br />4. Triton | https://github.com/NervanaSystems/maxas/wiki/SGEMM<br />https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ | -              |          |
| Benchmark       | Configure environment for <br />1. BLAS (CPU)<br />2. Intel MKL (CPU)<br />3. cutlass (GPU) |                                                              | 2 hours        |          |
| Visualization   | Performance Analysis                                         |                                                              | 2 hours        |          |

## Related Work

Optimizing General Matrix Multiplication (GEMM) requires tailored strategies for different hardware:

#### **1. Single-CPU Optimization (SIMD: SSE, AVX2, AVX512)**
- **Intel’s Optimization Manual** provides guidelines for SIMD programming (SSE/AVX2/AVX512), covering floating-point optimizations and memory alignment.
- **AVX512-Specific Challenges**: Wider registers (e.g., 512-bit) introduce memory bandwidth bottlenecks. Tutorials like *Optimizing DGEMM with AVX512F* address these issues through cache-aware blocking and instruction scheduling.
- **Eigen Library**: Implements AVX512-optimized GEMM kernels, offering insights into practical trade-offs (e.g., register pressure vs. parallelism).

#### **2. Multi-CPU Parallelization (OpenMP)**
- **Blocking + OpenMP**: Combining tiling techniques with OpenMP (e.g., `#pragma omp parallel for`) improves multi-core scaling. Discussions on platforms like Stack Overflow highlight empirical tuning for thread scheduling (static/dynamic).
- **BLIS**: A research introducing BLIS a new framework for rapid instantiation of the BLAS. It refines the GotoBLAS approach by exposing two additional loops around the inner kernel, enabling finer-grained parallelism.
  - ***micro-kernel***: Optimize the micro-kernel (a small, architecture-specific GEMM unit) for registers, simplifying porting across CPUs/GPUs.
  - ***Parallelism in shared caches***: Parallelizes five nested loops around the micro-kernel, targeting different memory hierarchies (L1/L2/L3 caches). 
  - ***Xeon Phi***: Uses hyperthreading and L2-cache blocking to mitigate bandwidth bottlenecks. Performance achieves matching Intel MKL.

#### **3. GPU Acceleration (CUDA/Metal)**
- **CUDA Optimization**: Key techniques include shared memory tiling, warp-level parallelism, and occupancy tuning.
- **Batched GEMM**: GPU-specific optimizations (e.g., kernel fusion) are explored in papers like *Performance, Design, and Autotuning of Batched GEMM for GPUs*.
- **Metal for Apple GPUs**: Leverages Apple’s Metal Performance Shaders (MPS) for GEMM, however, it's less extensive than CUDA.

#### **4. Reference**
- [1] T. M. Smith, R. v. d. Geijn, M. Smelyanskiy, J. R. Hammond and F. G. V. Zee, "Anatomy of High-Performance Many-Threaded Matrix Multiplication," 
- [2] Abdelfattah, A., Haidar, A., Tomov, S., Dongarra, J. (2016). Performance, Design, and Autotuning of Batched GEMM for GPUs. In: Kunkel, J., Balaji, P., Dongarra, J. (eds) High Performance Computing. ISC High Performance 2016. Lecture Notes in Computer Science(), vol 9697. Springer, Cham. https://doi.org/10.1007/978-3-319-41321-1_2

