# 18847 Project - Optimizing (DC)? GEMM on single CPU and multiple CPUs

## Key Points

1. Implementation for different types of data, for example, `double` , `float`, `complex` , and low-level integer. (generic programming)

2. Benchmark - GEMM or other? Or choose one problem using matrix multiplication, compare the performance using our implementation and its original implementation. 

3. **Single CPU - SIMD Extension includes *SSE, AVX2, AVX512*.**

   Multiple CPU - TBD (OpenMP?)

   **GPU - CUDA** 

4. Write assembly code on different chips such as Intel, AMD. (If required to compare with IntelMKL) 

   IntelMKL is the best software for dgemm on Intel chip. For AMD, BLAS? (BLAS is worse than IntelMKL)

5. Visualization for the algorithm process, and performance results.

 ## Synopsis - Jiayu Wang

If the format allows, start your proposal with a short summary, designed to convince the reviewer to read the rest of the proposal.

## Deliverables - Diac Liu

Include a brief, clear work breakdown structure with milestones and deadlines. Make sure to label deliverables as optional or required. You may want your plan to start by producing some kind of white paper, or planning the project in traditional Software Engineering style. It’s OK to include thinking time (“investigation”) in your work schedule. Deliverables should include investigation, coding and documentation.

## Related Work - Min Xiao

You should understand and communicate other people’s work that may be related to your own. Do your research, and make sure you understand how the project you are proposing fits into the target organization. Be sure to explain how the proposed work is different from similar related work.

## High-level Code Structure - Diac Liu


