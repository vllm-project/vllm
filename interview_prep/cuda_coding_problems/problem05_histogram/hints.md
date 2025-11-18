# Hints: Histogram

## Hint 1: Atomic Operations
```cuda
atomicAdd(&histogram[bin], 1);  // Thread-safe increment
```

## Hint 2: Privatization Strategy
```cuda
__shared__ int localHist[NUM_BINS];
// 1. Init local hist to 0
// 2. Accumulate to local hist (fast shared memory atomics)
// 3. Merge to global hist (fewer global atomics)
```

## Hint 3: Reducing Contention
- Shared memory atomics are ~20x faster than global
- Each block only does NUM_BINS global atomics (merge step)
- vs. every thread doing global atomics (naive)

## Hint 4: Warp Aggregation (Advanced)
Count within warp first, then one thread per warp does atomic add
