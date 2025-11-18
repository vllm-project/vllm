# Problem 14: Quantized INT8 Matrix Multiplication

**Difficulty:** Hard
**Estimated Time:** 60-75 minutes
**Tags:** Quantization, Mixed Precision, INT8, Tensor Cores, Inference

## Problem Statement

Implement a quantized matrix multiplication kernel that uses INT8 arithmetic for computation with FP32 dequantization. This is critical for efficient ML inference.

## Requirements

- Quantize FP32 matrices to INT8
- Perform INT8 matrix multiplication
- Dequantize back to FP32
- Use `dp4a` or `__imma` instructions if available
- Handle scale/zero-point quantization

## Quantization Formula

```
quantize: int8_val = round(fp32_val / scale) + zero_point
dequantize: fp32_val = (int8_val - zero_point) * scale
```

## Function Signature

```cuda
__global__ void quantizedGEMM(int8_t* A, int8_t* B, float* C,
                               float scale_A, float scale_B,
                               int M, int N, int K);
```

## Input/Output

**Input:**
- A: INT8 matrix (M × K)
- B: INT8 matrix (K × N)
- scale_A, scale_B: Quantization scales

**Output:**
- C: FP32 matrix (M × N)

## Success Criteria

- ✅ Correct quantization/dequantization
- ✅ INT8 computation
- ✅ Proper scaling
- ✅ Performance benefit over FP32
- ✅ Handles overflow/underflow
