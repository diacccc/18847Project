# 18847 Project - Optimizing (DC)? GEMM on single CPU and multiple CPUs

## Key Points

1. Implementation for different types of data, for example, `double` , `float`, `complex` , and low-level integer. (generic programming) OpenMP -> TBB

2. Benchmark - GEMM or other? Or choose one problem using matrix multiplication, compare the performance using our implementation and its original implementation. 

3. **Single CPU - SIMD Extension includes *SSE, AVX2, AVX512*.** Intrinsics 

   Multiple CPU - TBD (OpenMP?) pthread

   **GPU - CUDA** / Metal 

4. Write assembly code on different chips such as Intel, AMD. (If required to compare with IntelMKL) 

   IntelMKL is the best software for dgemm on Intel chip. For AMD, BLAS? (BLAS is worse than IntelMKL) 

5. Visualization for the algorithm process, and performance results.

 ## Synopsis - Jiayu Wang

This project aims to optimize the performance of General Matrix Multiplication (GEMM) by implementing SIMD (Single Instruction, Multiple Data) and multithreading on CPUs, as well as utilizing CUDA/METAL on GPUs. The main objectives are to enhance computational efficiency, reduce execution time, and achieve near-optimal performance. The methodology includes leveraging SIMD instructions and multithreading techniques on CPUs, and implementing CUDA/METAL kernels on GPUs. Expected outcomes include significant performance improvements in GEMM operations, validated through benchmarking and performance analysis by comparing the result with existing BLAS and Intel MKL benchmark. This project is to address critical performance bottlenecks in numerical computing and scientific applications, potentially benefiting a wide range of computational tasks。

## Deliverables - Diac Liu

| Feature           | Details                                                      | Resources                                                    | Estimated Time | Assignee |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | -------- |
| Generic Library   | Templated C++ GEMM implementation<br />1. double / float (Required)<br />2. Low-level float point f8e4m3 f8e5m2 (Optional)<br />3. Unit tests (Required) |                                                              | 4 hours        |          |
| Single CPU (SIMD) | 1. AVX2 (Required)<br />2. AVX512 (Required)                 | https://github.com/flame/how-to-optimize-gemm<br />https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf<br />https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html | -              |          |
| Multi-core        | 1. OpenMP (Linux) <br />2. TBB (MacOS)                       | https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf<br />https://www.openmp.org/ | -              |          |
| GPU               | 1. CUDA (Required)<br />2. Metal (MacOS)<br />3. OpenCL<br />4. Triton | https://zhuanlan.zhihu.com/p/435908830                       | -              |          |
| Benchmark         | Configure environment for <br />1. BLAS<br />2. Intel MKL    |                                                              | 2 hours        |          |
| Visualization     | Performance Analysis                                         |                                                              | 2 hours        |          |

## Related Work - Min Xiao

You should understand and communicate other people’s work that may be related to your own. Do your research, and make sure you understand how the project you are proposing fits into the target organization. Be sure to explain how the proposed work is different from similar related work.

## High-level Code Structure - Diac Liu

