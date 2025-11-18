# Hints: LayerNorm Kernel

## Hint 1: Formula
```
output = gamma * (x - mean) / sqrt(var + eps) + beta
```

## Hint 2: Two-Pass Approach
1. Pass 1: Compute mean
2. Pass 2: Compute variance using mean
3. Pass 3: Normalize and apply affine transform

## Hint 3: Numerical Stability
Use `rsqrtf()` for `1/sqrt(var + eps)` - faster and more stable

## Hint 4: Welford's Online Algorithm (Advanced)
Compute mean and variance in single pass:
```
for each value x:
    count++
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)
variance = M2 / count
```

## Hint 5: Memory Access
Each thread processes multiple elements with grid-stride loop
