# Hints: Softmax Kernel

## Hint 1: Numerical Stability
```cuda
// BAD: Can overflow with large values
output[i] = exp(x[i]) / sum(exp(x[j]))

// GOOD: Subtract max first
float max_x = max(x);
output[i] = exp(x[i] - max_x) / sum(exp(x[j] - max_x))
```

## Hint 2: Three-Pass Algorithm
Each block handles one row:
1. **Pass 1:** Find max value (warp reduce)
2. **Pass 2:** Compute exp(x - max) and sum (warp reduce)
3. **Pass 3:** Normalize by sum

## Hint 3: Warp Reduction
Use `__shfl_down_sync` for efficient warp-level max and sum:
```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
}
```

## Hint 4: Online Softmax (Advanced)
Compute in single pass using running statistics - more complex but faster
