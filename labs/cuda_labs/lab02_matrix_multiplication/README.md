# Lab 02: Matrix Multiplication - 2D Grids and Shared Memory

## Problem Statement

Implement CUDA kernels for matrix multiplication (C = A × B) with progressive optimizations. Matrix multiplication is fundamental to LLM inference, as it's the core operation in attention mechanisms, feed-forward networks, and projection layers.

**Relevance to LLM Inference:**
- Attention computation: Q×K^T and (Attention Weights)×V
- Linear transformations: X×W for every layer
- Represents >90% of compute in transformer models
- Understanding MM optimization directly translates to faster inference

## Learning Objectives

1. Master 2D thread indexing and grid configuration
2. Understand shared memory hierarchy and usage
3. Implement tiling strategies to reduce global memory access
4. Analyze memory access patterns and bank conflicts
5. Measure compute vs memory-bound performance transitions

## Prerequisites

- Completed Lab 01 (Vector Addition)
- Understanding of matrix operations
- Basic knowledge of memory hierarchies
- Familiarity with loop tiling concepts

## Estimated Time

3-4 hours

## Lab Structure

```
lab02_matrix_multiplication/
├── README.md          # This file
├── starter.cu         # Three implementations to complete
├── solution.cu        # Reference solutions
├── test.cu           # Validation and benchmarking
└── Makefile          # Build configuration
```

## Background: Matrix Multiplication

### Mathematical Definition
```
C[i][j] = Σ(k=0 to K-1) A[i][k] * B[k][j]
```

For matrices:
- A: M × K
- B: K × N
- C: M × N (output)

### Computational Complexity
- Operations: 2MNK (MNK multiplications + MNK additions)
- Memory: M×K + K×N + M×N elements read/written

### Performance Considerations
- **Arithmetic Intensity**: ~2K / 3 FLOP/byte (increases with K)
- Transition from memory-bound to compute-bound as K increases
- Ideal candidate for shared memory optimization

## Implementation Approaches

### Approach 1: Naive (Global Memory Only)
Each thread computes one output element:
- Read entire row of A (K elements)
- Read entire column of B (K elements)
- Compute dot product
- Write one element of C

**Limitations**: High global memory traffic, no data reuse

### Approach 2: Tiled with Shared Memory
Partition computation into tiles:
- Load tile of A into shared memory
- Load tile of B into shared memory
- Compute partial results using shared data
- Accumulate across tiles

**Benefits**: ~(block_size²) data reuse per tile

### Approach 3: Optimized (Multiple Elements per Thread)
Each thread computes multiple output elements:
- Increases register usage for partial results
- Better instruction-level parallelism
- Reduces shared memory loads per computed element

## Instructions

### Step 1: Implement Naive Kernel

Open `starter.cu` and complete `matrixMulNaive`:

**TODO 1**: Calculate row and column indices
**TODO 2**: Implement dot product computation
**TODO 3**: Write result to global memory

### Step 2: Implement Tiled Kernel

Complete `matrixMulTiled`:

**TODO 4**: Calculate tile indices
**TODO 5**: Collaborative tile loading to shared memory
**TODO 6**: Synchronize before computation
**TODO 7**: Compute using shared memory
**TODO 8**: Accumulate partial results across tiles

### Step 3: Implement Optimized Kernel

Complete `matrixMulOptimized`:

**TODO 9**: Compute multiple elements per thread (register tiling)
**TODO 10**: Vectorized loads where possible

### Step 4: Build and Test

```bash
# Build
make

# Run tests
make test

# Benchmark different sizes
./matrix_mul 512 512 512   # Small
./matrix_mul 1024 1024 1024  # Medium
./matrix_mul 4096 4096 4096  # Large

# Profile
make profile
```

## Expected Performance

**Configuration**: Square matrices (N×N), RTX 3080

| Size  | Naive (ms) | Tiled (ms) | Optimized (ms) | TFLOPS |
|-------|-----------|-----------|----------------|--------|
| 512   | 1.2       | 0.3       | 0.2            | 1.3    |
| 1024  | 9.5       | 2.1       | 1.4            | 1.5    |
| 2048  | 76.0      | 16.5      | 10.8           | 1.6    |
| 4096  | 608.0     | 131.0     | 84.0           | 1.6    |

**Speedup Factors**:
- Tiled vs Naive: ~4-5×
- Optimized vs Tiled: ~1.5-2×
- Optimized vs Naive: ~7-10×

## Profiling Instructions

### Nsight Systems - Timeline Analysis

```bash
# Profile all three kernels
nsys profile --stats=true ./solution 2048 2048 2048

# Focus on kernel duration
nsys profile --stats=true --force-overwrite=true \
     -o mm_profile ./solution 2048 2048 2048
```

**Look for**:
- Kernel execution time
- Memory transfer time
- Kernel launch overhead
- GPU utilization

### Nsight Compute - Detailed Kernel Metrics

```bash
# Full metrics
ncu --set full ./solution 2048 2048 2048

# Memory-specific metrics
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
dram__bytes_read.sum,dram__bytes_write.sum \
./solution 2048 2048 2048

# Shared memory metrics
ncu --metrics lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
./solution 2048 2048 2048

# Occupancy and compute
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
./solution 2048 2048 2048
```

## Common Mistakes and Debugging

### 1. Incorrect 2D Indexing

**Mistake**: Row-major vs column-major confusion
```cuda
// WRONG
int row = threadIdx.x + blockIdx.x * blockDim.x;
int col = threadIdx.y + blockIdx.y * blockDim.y;

// CORRECT (depends on your layout)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### 2. Missing Synchronization

**Mistake**: Computing before shared memory is fully loaded
```cuda
// WRONG
__shared__ float As[TILE_SIZE][TILE_SIZE];
As[ty][tx] = A[...];  // Load
float sum = As[ty][0] * Bs[0][tx];  // Use immediately - RACE!

// CORRECT
As[ty][tx] = A[...];
__syncthreads();  // Wait for all threads
float sum = As[ty][0] * Bs[0][tx];
```

### 3. Shared Memory Bank Conflicts

**Issue**: Multiple threads accessing same bank
```cuda
// POTENTIAL CONFLICT (stride = blockDim.x)
__shared__ float data[TILE_SIZE][TILE_SIZE];
float value = data[threadIdx.x][threadIdx.y];  // Column access

// BETTER (if accessing columns frequently, pad)
__shared__ float data[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
```

### 4. Boundary Conditions

**Mistake**: Not handling non-tile-aligned matrices
```cuda
// Check boundaries when loading tiles
if (row < M && k < K) {
    As[ty][tx] = A[row * K + k];
} else {
    As[ty][tx] = 0.0f;  // Pad with zeros
}
```

### 5. Incorrect Accumulation

**Mistake**: Not accumulating across all tiles
```cuda
// WRONG
for (int t = 0; t < numTiles; t++) {
    float sum = 0.0f;  // Reset each iteration - WRONG!
    // ... computation
}

// CORRECT
float sum = 0.0f;  // Initialize once
for (int t = 0; t < numTiles; t++) {
    // ... accumulate into sum
}
```

## Performance Analysis

### Memory Access Patterns

**Naive Kernel**:
- Global memory accesses: 2K per output element
- Total: 2MNK global reads
- No reuse between threads

**Tiled Kernel**:
- Global memory accesses: 2(M×K + K×N) / TILE_SIZE
- Shared memory reuse: ~TILE_SIZE²
- Reduction factor: ~TILE_SIZE

**Optimized Kernel**:
- Further reduces shared memory loads per output
- Better register utilization
- Improved instruction-level parallelism

### Roofline Model Analysis

For N×N×N matrix multiplication:

```
Arithmetic Intensity (AI) = 2N³ / (3N² × 4 bytes) = 2N/3 FLOP/byte
```

- Small N (<100): Memory-bound
- Medium N (100-1000): Transitioning
- Large N (>1000): Compute-bound

**Optimization Strategy**:
- Memory-bound: Reduce traffic (tiling, reuse)
- Compute-bound: Increase throughput (ILP, occupancy)

## Optimization Challenges

### Challenge 1: Rectangular Matrices
Extend to handle non-square matrices (M ≠ K ≠ N):
- Adjust tiling strategy
- Handle boundary conditions
- Test with various shapes

### Challenge 2: Bank Conflict Elimination
Modify tiled kernel to eliminate bank conflicts:
- Use padding in shared memory
- Analyze with Nsight Compute
- Measure performance improvement

### Challenge 3: Double Buffering
Overlap computation and memory loads:
- Use two shared memory buffers
- Load next tile while computing current
- Requires careful synchronization

### Challenge 4: Tensor Core Integration
Use Tensor Cores (Volta+):
- Implement using WMMA API
- Compare performance to CUDA cores
- Measure achievable TFLOPS

### Challenge 5: Batched Matrix Multiplication
Extend to batch of matrices:
- Use 3D grid
- Optimize for memory coalescing
- Compare to cuBLAS

### Challenge 6: Mixed Precision
Implement FP16 inputs, FP32 accumulation:
- Relevant for LLM inference
- Use `__half` types
- Measure accuracy vs performance tradeoff

## Key Takeaways

1. **2D Indexing**: `row = blockIdx.y * blockDim.y + threadIdx.y`
2. **Tiling**: Essential for reusing data in fast memory
3. **Shared Memory**: Reduces global memory traffic by ~TILE_SIZE
4. **Synchronization**: `__syncthreads()` after loading, before use
5. **Boundary Checks**: Handle non-tile-aligned matrices
6. **Arithmetic Intensity**: Increases with matrix size, transitions to compute-bound

## Real-World Connection

In LLM inference:
- **Attention Q×K^T**: Often small batch, large sequence → optimize for this
- **Attention×V**: Different dimensions, same operation
- **FFN layers**: Typically compute-bound (large K)
- **Batching**: Multiple sequences → batched MM

Understanding these patterns helps optimize real transformer models.

## Next Steps

After completing this lab:
1. Compare your implementation with cuBLAS (`cublasSgemm`)
2. Profile different matrix shapes representative of LLM layers
3. Experiment with Tensor Cores (if available)
4. Proceed to Lab 03: Reduction Operations

## References

- [CUDA C Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Tensor Cores WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

## Troubleshooting

**Problem**: Results incorrect
- Check matrix indexing (row-major vs column-major)
- Verify synchronization points
- Test with small matrices and print intermediate values

**Problem**: Shared memory errors
- Check tile size doesn't exceed shared memory limit
- Verify `__syncthreads()` is not in divergent code

**Problem**: Poor performance
- Profile with Nsight Compute
- Check for bank conflicts
- Verify occupancy is reasonable (>50%)

**Problem**: Slower than expected
- Compare with cuBLAS baseline
- Check if compute-bound or memory-bound
- Verify compiler optimizations enabled (`-O3`)
