# Hints: Quantized INT8 GEMM

## Hint 1: Why INT8?
- 4x less memory than FP32
- 4x faster on modern GPUs (Tensor Cores, dp4a)
- Critical for inference optimization

## Hint 2: Quantization
```cuda
int8_val = clamp(round(fp32_val / scale) + zero_point, -128, 127)
```

## Hint 3: Accumulation
Use INT32 for accumulation to avoid overflow:
```cuda
int32_t sum = 0;
sum += (int32_t)a * (int32_t)b;  // INT8 × INT8 → INT32
```

## Hint 4: Dequantization
```cuda
fp32_result = (int32_t)sum * scale_A * scale_B;
```

## Hint 5: DP4A Instruction
```cuda
__dp4a(a_vec, b_vec, c);  // 4× INT8 multiplies + accumulate
// Requires SM_61+ (Pascal or later)
```
