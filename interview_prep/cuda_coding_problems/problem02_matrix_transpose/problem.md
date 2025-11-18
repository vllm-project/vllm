# Problem 2: Optimized Matrix Transpose

**Difficulty:** Easy
**Estimated Time:** 30-45 minutes
**Tags:** Memory Access Patterns, Shared Memory, Bank Conflicts

## Problem Statement

Implement an efficient CUDA kernel for transposing a matrix. Your solution should optimize memory access patterns to achieve high bandwidth utilization and avoid shared memory bank conflicts.

## Requirements

Write a CUDA kernel that:
1. Transposes an M×N matrix (rows ↔ columns)
2. Uses shared memory tiles for optimization
3. Achieves coalesced memory access for both reads and writes
4. Avoids shared memory bank conflicts
5. Handles non-square matrices and non-tile-aligned dimensions

## Function Signature

```cuda
__global__ void transposeKernel(float* input, float* output, int rows, int cols);
void matrixTranspose(float* h_input, float* h_output, int rows, int cols);
```

## Input/Output Specifications

**Input:**
- `input`: M×N matrix in row-major order
- `rows`: Number of rows (M)
- `cols`: Number of columns (N)

**Output:**
- `output`: N×M transposed matrix in row-major order

**Constraints:**
- 1 ≤ rows, cols ≤ 8192
- Matrix elements are 32-bit floats
- Must use shared memory with tile size 32×32

## Examples

### Example 1: Small Matrix
```
Input (3×4):
1  2  3  4
5  6  7  8
9  10 11 12

Output (4×3):
1  5  9
2  6  10
3  7  11
4  8  12
```

### Example 2: Square Matrix
```
Input (2×2):
1 2
3 4

Output (2×2):
1 3
2 4
```

### Example 3: Single Row
```
Input (1×5):
1 2 3 4 5

Output (5×1):
1
2
3
4
5
```

## Key Considerations

1. **Coalesced Access:** Read/write consecutive elements for coalescing
2. **Bank Conflicts:** Add padding to shared memory to avoid conflicts
3. **Tile Boundaries:** Handle edge cases where matrix isn't tile-aligned
4. **Performance:** Should achieve >80% of memory bandwidth

## Common Pitfalls

- Uncoalesced global memory writes in naive implementation
- Shared memory bank conflicts (all threads access same bank)
- Not handling non-tile-aligned dimensions
- Incorrect indexing for row-major to column-major conversion
- Race conditions in shared memory

## Follow-Up Questions

1. Why does naive transpose have poor performance?
2. How does padding eliminate bank conflicts?
3. What's the performance difference between transpose and copy?
4. How would you handle 3D tensor transpose?
5. Can you use texture memory for optimization?

## Success Criteria

- ✅ Correct transpose for all test cases
- ✅ Coalesced memory access patterns
- ✅ No bank conflicts in shared memory
- ✅ Handles non-square and non-aligned matrices
- ✅ Performance within 20% of cuBLAS geam
