# Progressive Hints: Parallel Reduction

## Hint 1: Understanding the Problem (Free)

**Think about:**
- How would you sum an array sequentially on CPU?
- Can you parallelize this by having threads work on different parts?
- What's the challenge with having each thread write to a single sum variable?

**Key Insight:** You need a tree-based reduction to avoid race conditions and maximize parallelism.

---

## Hint 2: Basic Structure (After 5 minutes)

**Algorithm Outline:**
1. Each thread loads one (or more) elements
2. Threads cooperate to reduce within a block using shared memory
3. Each block produces one partial sum
4. Final reduction combines block results

**Think about:** Why use shared memory instead of global memory for reduction?

---

## Hint 3: Reduction Tree Pattern (After 10 minutes)

**Classic Reduction Pattern:**
```
Initial: [a0, a1, a2, a3, a4, a5, a6, a7]
Step 1:  [a0+a1, a2+a3, a4+a5, a6+a7]  (4 sums)
Step 2:  [a0+a1+a2+a3, a4+a5+a6+a7]    (2 sums)
Step 3:  [a0+a1+a2+a3+a4+a5+a6+a7]     (1 sum)
```

**Questions:**
- How many steps for N elements?
- Which threads are active at each step?
- When do you need `__syncthreads()`?

---

## Hint 4: Avoiding Bank Conflicts (After 15 minutes)

**Problem with Interleaved Addressing:**
```cuda
// BAD: Creates bank conflicts
unsigned int s = 1;
while (s < blockDim.x) {
    if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    s *= 2;
}
```

**Why is this bad?**
- Warp divergence (only some threads in warp are active)
- Bank conflicts in shared memory

**Better Approach:** Use sequential addressing instead!

---

## Hint 5: Sequential Addressing (After 20 minutes)

**Solution:**
```cuda
// GOOD: Sequential addressing
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**Why is this better?**
- Adjacent threads (in same warp) are active
- No bank conflicts
- Better memory access pattern

---

## Hint 6: Multiple Elements Per Thread (After 25 minutes)

**Optimization:** Load multiple elements per thread to reduce blocks needed

```cuda
unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
float sum = 0.0f;
while (idx < n) {
    sum += input[idx];
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }
    idx += gridSize;
}
sdata[tid] = sum;
```

**Benefits:**
- Fewer blocks needed
- Better memory coalescing
- Reduces overhead of final reduction

---

## Hint 7: Warp-Level Primitives (After 30 minutes)

**Advanced Optimization:** Use warp shuffle for final warp

```cuda
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**When to use:**
- Last 32 elements (one warp)
- No `__syncthreads()` needed
- Faster than shared memory

---

## Hint 8: Handling Arbitrary Sizes (After 35 minutes)

**Edge Cases to Handle:**
1. **n < block_size:** May only need one block
2. **n not power of 2:** Add bounds checks
3. **n = 0 or n = 1:** Handle in host code
4. **Very large n:** May need multi-level reduction

**Example Check:**
```cuda
if (idx + blockDim.x < n) {
    sum += input[idx + blockDim.x];
}
```

---

## Hint 9: Complete Kernel Structure (If still stuck)

**Template:**
```cuda
__global__ void reductionKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    // 1. Load + first reduction (grid-stride loop)
    // TODO: Load multiple elements per thread

    // 2. Store to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // 3. Reduction in shared memory
    // TODO: Sequential addressing reduction

    // 4. Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

---

## Hint 10: Common Mistakes to Avoid

1. **Forgetting `__syncthreads()`** after writing to shared memory
2. **Using `__syncthreads()` inside divergent branch** (undefined behavior)
3. **Not handling non-power-of-2 sizes**
4. **Incorrect grid/block size calculations**
5. **Not checking for `idx < n` in kernel**
6. **Memory leaks** (not freeing device memory)
7. **Not checking CUDA errors**

---

## Debugging Checklist

If your solution isn't working:

- [ ] Print intermediate results from each block
- [ ] Try with small, known input (e.g., [1,2,3,4])
- [ ] Check if block size is power of 2
- [ ] Verify `__syncthreads()` is called at right places
- [ ] Check array bounds in all memory accesses
- [ ] Use `cuda-memcheck` to find race conditions
- [ ] Verify shared memory size is correct

---

## Performance Tips

**Expected Performance:**
- Should achieve >80% of memory bandwidth
- Typical: 100-300 GB/s on modern GPUs
- Compare with cuBLAS `cublasSasum` for reference

**If slow:**
- Check occupancy with `nvprof --metrics achieved_occupancy`
- Verify coalesced memory access
- Try different block sizes (128, 256, 512)
- Profile with `nvprof` or Nsight Compute
