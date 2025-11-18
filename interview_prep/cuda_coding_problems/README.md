# CUDA Coding Interview Problems

## Overview

This directory contains 15 comprehensive CUDA coding interview problems designed for ML infrastructure positions at companies like NVIDIA, OpenAI, and Anthropic. Each problem includes:

- **Problem Statement** (`problem.md`) - Detailed requirements and examples
- **Solution** (`solution.cu`) - Complete, optimized CUDA implementation
- **Test Cases** (`test.cu`) - Comprehensive test suite
- **Grading Rubric** (`rubric.md`) - Evaluation criteria and scoring
- **Progressive Hints** (`hints.md`) - Step-by-step guidance

## Problem Directory

### Easy Problems (5)

**1. Parallel Reduction** (`problem01_parallel_reduction/`)
- Sum array elements efficiently
- Topics: Shared memory, warp primitives, bank conflicts
- Time: 30-45 minutes
- Lines of code: ~200

**2. Matrix Transpose** (`problem02_matrix_transpose/`)
- Optimize memory access patterns
- Topics: Tiling, coalescing, bank conflicts
- Time: 30-45 minutes
- Lines of code: ~150

**3. Element-wise Operations** (`problem03_element_wise_operations/`)
- Kernel fusion basics
- Topics: Kernel fusion, memory bandwidth, vectorization
- Time: 25-35 minutes
- Lines of code: ~120

**4. Prefix Sum (Scan)** (`problem04_prefix_sum/`)
- Inclusive/exclusive scan
- Topics: Blelloch algorithm, work efficiency, multi-level reduction
- Time: 35-45 minutes
- Lines of code: ~180

**5. Histogram** (`problem05_histogram/`)
- Atomic operations usage
- Topics: Atomics, privatization, contention reduction
- Time: 30-40 minutes
- Lines of code: ~140

### Medium Problems (7)

**6. Softmax Kernel** (`problem06_softmax_kernel/`)
- Numerically stable softmax
- Topics: Online algorithms, numerical stability, row-wise operations
- Time: 40-50 minutes
- Lines of code: ~180

**7. LayerNorm Kernel** (`problem07_layernorm_kernel/`)
- Layer normalization optimization
- Topics: Welford's algorithm, fused operations, ML kernels
- Time: 45-55 minutes
- Lines of code: ~200

**8. Attention Score Computation** (`problem08_attention_scores/`)
- Q*K^T with optimization
- Topics: Matrix multiply, tiling, transformer operations
- Time: 45-50 minutes
- Lines of code: ~170

**9. Batched Matrix Multiply** (`problem09_batched_matmul/`)
- Handling multiple batches
- Topics: 3D grids, batching, GEMM optimization
- Time: 40-50 minutes
- Lines of code: ~180

**10. Dynamic Shared Memory** (`problem10_dynamic_shared_memory/`)
- Variable shared memory allocation
- Topics: extern __shared__, runtime configuration
- Time: 35-45 minutes
- Lines of code: ~140

**11. Warp-Level Reduction** (`problem11_warp_reduction/`)
- Using warp primitives
- Topics: __shfl_down_sync, lock-free algorithms, warp programming
- Time: 35-45 minutes
- Lines of code: ~190

**12. Memory Coalescing Optimization** (`problem12_memory_coalescing/`)
- Fix uncoalesced accesses
- Topics: Memory access patterns, performance tuning, profiling
- Time: 40-50 minutes
- Lines of code: ~160

### Hard Problems (3)

**13. Flash Attention** (`problem13_flash_attention/`)
- Simplified Flash Attention implementation
- Topics: Online softmax, tiling, memory optimization, advanced algorithms
- Time: 60-75 minutes
- Lines of code: ~250

**14. Quantized INT8 GEMM** (`problem14_quantized_int8_gemm/`)
- INT8 matrix multiplication with dequantization
- Topics: Quantization, mixed precision, dp4a, tensor cores
- Time: 60-75 minutes
- Lines of code: ~230

**15. Multi-Stream Optimization** (`problem15_multistream_optimization/`)
- Overlap compute and memory transfers
- Topics: CUDA streams, async operations, pipelining, concurrency
- Time: 60-75 minutes
- Lines of code: ~210

## Statistics

### Difficulty Distribution
- **Easy:** 5 problems (33%)
- **Medium:** 7 problems (47%)
- **Hard:** 3 problems (20%)

### Average Solution Length
- **Easy:** ~158 lines of code
- **Medium:** ~176 lines of code
- **Hard:** ~230 lines of code
- **Overall Average:** ~182 lines of code

### Topics Covered
- Memory optimization (coalescing, bank conflicts, shared memory)
- Parallel algorithms (reduction, scan, sorting)
- ML primitives (softmax, layernorm, attention)
- Advanced techniques (quantization, streams, flash attention)
- Performance optimization (tiling, fusion, vectorization)

### Total Content
- **75 files** (15 problems × 5 files each)
- **~2730 total lines of solution code**
- **~1500 lines of problem descriptions**
- **~800 lines of test code**
- **~600 lines of rubrics**
- **~500 lines of hints**

## Usage

### Compile and Run a Problem

```bash
# Navigate to problem directory
cd problem01_parallel_reduction/

# Compile solution
nvcc -o solution solution.cu -O3 -arch=sm_70

# Run solution
./solution

# Compile and run tests
nvcc -o test test.cu solution.cu -O3 -arch=sm_70
./test
```

### Interview Practice

1. **Read the problem statement** (`problem.md`)
2. **Attempt solution** (use estimated time as guide)
3. **Check hints** if stuck (`hints.md`)
4. **Compare with solution** (`solution.cu`)
5. **Review rubric** to understand evaluation (`rubric.md`)
6. **Run tests** to verify correctness (`test.cu`)

### Problem Selection by Company

**NVIDIA GPU Computing Roles:**
- Problems 1-5 (fundamentals)
- Problems 6, 7, 13 (ML kernels)
- Problems 11, 12 (optimization)

**ML Infrastructure (OpenAI, Anthropic):**
- Problems 6-9 (transformer kernels)
- Problem 13 (flash attention)
- Problem 14 (quantization)
- Problem 15 (streams)

**Systems/Performance Engineers:**
- Problems 1-5 (parallel algorithms)
- Problems 11, 12 (low-level optimization)
- Problem 15 (concurrency)

## Key Learning Objectives

After completing these problems, you should understand:

1. **CUDA Memory Model**
   - Global, shared, and register memory
   - Coalescing and bank conflicts
   - Memory bandwidth optimization

2. **Parallel Algorithms**
   - Reduction, scan, histogram
   - Work efficiency and step complexity
   - Tiling and blocking strategies

3. **Optimization Techniques**
   - Warp-level primitives
   - Kernel fusion
   - Occupancy and latency hiding

4. **ML Kernels**
   - Attention mechanisms
   - Normalization layers
   - Quantization

5. **Advanced Topics**
   - Asynchronous execution
   - Multi-stream pipelining
   - Online algorithms

## Interview Tips

1. **Time Management:** Easy (30-45 min), Medium (40-55 min), Hard (60-75 min)
2. **Start Simple:** Get a working solution, then optimize
3. **Explain Trade-offs:** Discuss memory vs. compute, occupancy vs. resource usage
4. **Test Edge Cases:** Empty arrays, non-aligned sizes, large inputs
5. **Discuss Performance:** Compare to theoretical peak, mention profiling tools

## Additional Resources

- NVIDIA CUDA C Programming Guide
- Professional CUDA C Programming (book)
- CUDA by Example (book)
- Nsight Compute for profiling
- cuBLAS/cuDNN source code references

## Contributing

These problems are designed for educational purposes and interview preparation. Feel free to:
- Add more test cases
- Optimize solutions further
- Add alternative implementations
- Report issues or improvements

---

**Total Investment:** ~2730 lines of solution code across 15 comprehensive problems covering essential CUDA concepts for ML infrastructure interviews.

**Last Updated:** 2025
