# Problem 7: Layer Normalization Kernel

**Difficulty:** Medium
**Estimated Time:** 45-55 minutes
**Tags:** ML Kernels, Online Algorithms, Numerical Stability

## Problem Statement

Implement an optimized LayerNorm kernel in CUDA. Given a 2D tensor (batch_size × hidden_dim), normalize each row to have mean=0 and variance=1, then apply learned scale and bias parameters.

## Requirements

- Row-wise normalization: `output = gamma * (x - mean) / sqrt(var + eps) + beta`
- Numerically stable variance computation
- Fused operations (compute mean, var, normalize in minimal passes)
- Handle arbitrary dimensions
- Use Welford's online algorithm for numerical stability

## Function Signature

```cuda
__global__ void layerNormKernel(float* input, float* output, float* gamma,
                                float* beta, int rows, int cols, float eps);
```

## Input/Output

**Input:**
- `input`: Matrix (rows × cols)
- `gamma`, `beta`: Learned parameters (length cols)
- `eps`: Small constant for numerical stability (1e-5)

**Output:**
- `output`: Normalized matrix (rows × cols)

## Algorithm

For each row:
1. Compute mean: `μ = sum(x) / N`
2. Compute variance: `σ² = sum((x - μ)²) / N`
3. Normalize: `y = gamma * (x - μ) / sqrt(σ² + eps) + beta`

## Example

```
Input: [[1, 2, 3]]
gamma: [1, 1, 1]
beta: [0, 0, 0]
eps: 1e-5

mean = 2, var = 2/3
Output: [[-1.225, 0, 1.225]]  # standardized
```

## Success Criteria

- ✅ Correct normalization (mean≈0, std≈1 before scale/bias)
- ✅ Numerically stable
- ✅ Fused computation (2-pass or online)
- ✅ Efficient memory access
