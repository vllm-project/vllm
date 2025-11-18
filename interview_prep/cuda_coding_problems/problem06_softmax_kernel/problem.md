# Problem 6: Numerically Stable Softmax Kernel

**Difficulty:** Medium
**Estimated Time:** 40-50 minutes
**Tags:** ML Kernels, Numerical Stability, Row-wise Operations

## Problem Statement

Implement a numerically stable softmax kernel in CUDA. Given a 2D matrix (batch_size × features), compute softmax along the feature dimension for each row.

## Requirements

- Numerically stable (subtract max before exp)
- Row-wise softmax: `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))`
- Handle arbitrary matrix dimensions
- Use shared memory for reduction operations
- Optimize for memory bandwidth

## Function Signature

```cuda
__global__ void softmaxKernel(float* input, float* output, int rows, int cols);
void softmax(float* h_input, float* h_output, int rows, int cols);
```

## Input/Output

**Input:**
- `input`: Matrix of shape (rows × cols)
- Each row is an independent softmax operation

**Output:**
- `output`: Softmax probabilities (rows × cols)
- Each row sums to 1.0

## Algorithm

For each row:
1. Find max value: `m = max(x)`
2. Compute exp(x - m): `exp_x[i] = exp(x[i] - m)`
3. Sum exponentials: `sum_exp = sum(exp_x)`
4. Normalize: `output[i] = exp_x[i] / sum_exp`

## Example

```
Input: [[1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0]]

Output: [[0.09, 0.24, 0.67],   // exp(1-3) / Z, exp(2-3) / Z, exp(3-3) / Z
         [0.33, 0.33, 0.33]]   // uniform
```

## Success Criteria

- ✅ Numerically stable (handles large/small values)
- ✅ Each row sums to 1.0 (within epsilon)
- ✅ Handles arbitrary dimensions
- ✅ Uses shared memory for reductions
- ✅ Efficient memory access patterns
