# Hints: Element-wise Operations

## Hint 1: Kernel Fusion Benefit
Instead of 3 separate kernels (multiply, add, add constant), fuse into one kernel:
- Reduces memory traffic from 3 reads + 3 writes to 3 reads + 1 write
- Better cache utilization

## Hint 2: Grid-Stride Loop Pattern
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = idx; i < n; i += stride) {
    D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma;
}
```

## Hint 3: Vectorization (Advanced)
Use float4 to load/store 4 elements at once:
```cuda
float4 a = reinterpret_cast<float4*>(A)[i];
// Process a.x, a.y, a.z, a.w
```
Requires proper alignment and n % 4 == 0.
