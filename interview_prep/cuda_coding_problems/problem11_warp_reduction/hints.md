# Hints: Warp Reduction

## Hint 1: Shuffle Down Pattern
```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Lane 0 now has sum of all 32 lanes
```

## Hint 2: Synchronization Mask
`0xffffffff` means all 32 threads in warp participate

## Hint 3: Why Shuffles?
- Faster than shared memory
- No bank conflicts
- Lock-free within warp
- Modern GPUs optimized for this

## Hint 4: Block-Level
Use shared memory only to combine warp results
