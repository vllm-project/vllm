# Problem 10: Dynamic Shared Memory Allocation

**Difficulty:** Medium
**Estimated Time:** 35-45 minutes
**Tags:** Shared Memory, Dynamic Allocation, Configuration

## Problem Statement

Implement a matrix transpose kernel using dynamically allocated shared memory. The tile size should be configurable at kernel launch time rather than compile time.

## Requirements

- Use `extern __shared__` for dynamic allocation
- Support variable tile sizes (16, 32, 64, etc.)
- Pass shared memory size at kernel launch
- Handle padding to avoid bank conflicts
- Works with non-square matrices

## Function Signature

```cuda
__global__ void dynamicTransposeKernel(float* input, float* output,
                                       int rows, int cols, int tileSize);
```

## Example Launch

```cuda
int tileSize = 32;
size_t sharedMemSize = (tileSize * (tileSize + 1)) * sizeof(float);
dynamicTransposeKernel<<<grid, block, sharedMemSize>>>(
    input, output, rows, cols, tileSize);
```

## Success Criteria

- ✅ Correctly transposes matrices
- ✅ Uses dynamic shared memory
- ✅ Works with multiple tile sizes
- ✅ Bank conflict avoidance with padding
- ✅ Handles non-tile-aligned dimensions
