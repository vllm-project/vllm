# Lab 01: Vector Addition - CUDA Fundamentals

## Problem Statement

Implement a basic CUDA kernel to perform element-wise addition of two vectors. This foundational operation is crucial for understanding how GPU parallelism works and is a building block for more complex operations in LLM inference, such as adding positional encodings or residual connections.

**Relevance to LLM Inference:**
- Element-wise operations are ubiquitous in transformer models (bias addition, residual connections)
- Understanding thread indexing is essential for implementing attention mechanisms
- Memory transfer patterns learned here apply to loading model weights and activations

## Learning Objectives

1. Understand CUDA kernel launch syntax and thread hierarchy (grid, blocks, threads)
2. Master 1D thread indexing calculations for parallel data processing
3. Learn proper memory allocation and transfer between host and device
4. Implement error checking for CUDA API calls
5. Measure and compare GPU vs CPU performance

## Prerequisites

- Basic C/C++ programming knowledge
- Understanding of arrays and pointers
- Familiarity with compilation process
- CUDA Toolkit installed (tested with CUDA 11.0+)

## Estimated Time

2-3 hours

## Lab Structure

```
lab01_vector_addition/
├── README.md          # This file
├── starter.cu         # Your implementation goes here
├── solution.cu        # Reference solution
├── test.cu           # Validation and benchmarking
└── Makefile          # Build configuration
```

## Instructions

### Step 1: Understanding the Problem

You need to compute: `C[i] = A[i] + B[i]` for vectors of size N.

**Sequential approach (CPU):**
```c
for(int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

**Parallel approach (GPU):**
Each thread computes one element:
```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < N) {
    C[i] = A[i] + B[i];
}
```

### Step 2: Implementation Tasks

Open `starter.cu` and complete the following TODOs:

1. **TODO 1**: Implement the `vectorAdd` kernel
   - Calculate global thread index
   - Add boundary check
   - Perform the addition

2. **TODO 2**: Allocate device memory for three vectors

3. **TODO 3**: Copy input vectors from host to device

4. **TODO 4**: Configure and launch the kernel
   - Calculate grid dimensions
   - Choose appropriate block size

5. **TODO 5**: Copy result from device to host

6. **TODO 6**: Free device memory

### Step 3: Build and Run

```bash
# Build
make

# Run with default size (1M elements)
./vector_add

# Run with custom size
./vector_add 10000000

# Clean
make clean
```

### Step 4: Validation

The program will:
1. Initialize random vectors on CPU
2. Run CPU version for correctness check
3. Run GPU version
4. Compare results (should match within floating-point tolerance)
5. Report performance metrics

## Expected Performance

**Configuration:**
- Vector size: 1M elements (4 MB per vector)
- Block size: 256 threads
- Grid size: Calculated to cover all elements

**Expected Results (on RTX 3080):**
- CPU Time: ~2-3 ms
- GPU Kernel Time: ~0.1 ms
- GPU Total Time (including transfers): ~1.5 ms
- Speedup: ~2x (transfer overhead dominates for small workloads)

**For larger vectors (100M elements):**
- Speedup: ~10-20x (computation dominates, amortizing transfer costs)

## Profiling Instructions

### Using Nsight Systems (nsys)

```bash
# Profile the application
nsys profile --stats=true ./vector_add 10000000

# Generate detailed report
nsys profile -o vector_add_profile ./vector_add 10000000

# View in Nsight Systems GUI
nsys-ui vector_add_profile.nsys-rep
```

**What to look for:**
- Kernel execution time
- Memory transfer times (HtoD and DtoH)
- GPU utilization
- Memory bandwidth utilization

### Using Nsight Compute (ncu)

```bash
# Profile kernel metrics
ncu --set full ./vector_add 10000000

# Focus on memory metrics
ncu --metrics dram_throughput,l2_cache_throughput ./vector_add
```

## Common Mistakes and Debugging Tips

### 1. Off-by-One Errors
**Mistake:** Not checking if `threadIdx >= N`
```cuda
// WRONG
C[i] = A[i] + B[i];  // May access out of bounds

// CORRECT
if(i < N) {
    C[i] = A[i] + B[i];
}
```

### 2. Incorrect Grid Size Calculation
**Mistake:** Integer division truncation
```cuda
// WRONG - misses last block if N % blockSize != 0
int numBlocks = N / blockSize;

// CORRECT - ceiling division
int numBlocks = (N + blockSize - 1) / blockSize;
```

### 3. Missing Error Checking
```cuda
// Always check CUDA calls
cudaError_t err = cudaMalloc(&d_A, size);
if(err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

### 4. Memory Leak
```cuda
// Don't forget to free!
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

### 5. Synchronization Issues
```cuda
// Ensure kernel completes before copying results
cudaDeviceSynchronize();  // Often implicit in cudaMemcpy
```

## Performance Analysis

### Memory Bandwidth

**Theoretical Peak (RTX 3080):** ~760 GB/s

**Effective Bandwidth Calculation:**
```
Total Data = 3 * N * sizeof(float)  // Read A, B, write C
Effective BW = Total Data / Kernel Time
```

For vector addition, you should achieve 60-80% of peak bandwidth.

### Optimization Questions

1. **Why is speedup limited for small vectors?**
   - Memory transfer overhead
   - Kernel launch overhead
   - Not enough work to saturate GPU

2. **What's the optimal block size?**
   - Typically 128-512 threads
   - Must be multiple of 32 (warp size)
   - Experiment with different values

3. **Can we eliminate transfers?**
   - Use unified memory (cudaMallocManaged)
   - Keep data on GPU for subsequent operations

## Optimization Challenges

### Challenge 1: Block Size Tuning
Modify the code to test different block sizes (64, 128, 256, 512, 1024) and plot performance.

**Expected insight:** Block size impact is minimal for memory-bound kernels.

### Challenge 2: Unified Memory
Replace explicit memory management with `cudaMallocManaged`. Compare:
- Code simplicity
- Performance (with/without prefetching)

### Challenge 3: Streams for Overlap
Use CUDA streams to overlap computation with data transfer:
- Divide work into chunks
- Transfer and compute chunk-by-chunk

### Challenge 4: Multiple Operations
Extend to compute: `D[i] = A[i] + B[i] * C[i]`
- Compare launching separate kernels vs fused kernel
- Analyze bandwidth utilization

## Key Takeaways

1. **Thread Indexing**: `i = blockIdx.x * blockDim.x + threadIdx.x`
2. **Boundary Checks**: Always check `i < N` to avoid out-of-bounds access
3. **Grid Sizing**: Use ceiling division: `(N + blockSize - 1) / blockSize`
4. **Error Handling**: Check all CUDA API return values
5. **Memory Management**: Allocate, copy, use, copy back, free
6. **Performance**: Memory-bound operations are limited by bandwidth, not compute

## Next Steps

After completing this lab:
1. Experiment with different vector sizes and block sizes
2. Profile with nsys/ncu to understand performance bottlenecks
3. Try the optimization challenges
4. Move on to Lab 02: Matrix Multiplication (2D indexing)

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

## Troubleshooting

**Problem:** "no CUDA-capable device detected"
- Check: `nvidia-smi`
- Verify CUDA installation

**Problem:** Compilation errors
- Check CUDA Toolkit version: `nvcc --version`
- Ensure compute capability matches your GPU

**Problem:** Incorrect results
- Add debug prints inside kernel (use `printf` - but sparingly!)
- Verify CPU implementation is correct
- Check for race conditions or out-of-bounds access

**Problem:** Poor performance
- Profile with nsys/ncu
- Check if kernel is actually running on GPU
- Verify memory transfers are not dominating
