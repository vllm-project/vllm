# Hints: Memory Coalescing

## Hint 1: What is Coalescing?
Consecutive threads accessing consecutive memory addresses → single memory transaction

## Hint 2: Uncoalesced Pattern
```cuda
output[idx] = input[idx * stride];  // Thread N accesses N*stride
// Threads 0,1,2,3 access 0, stride, 2*stride, 3*stride (not consecutive!)
```

## Hint 3: Coalesced Pattern
Use shared memory:
1. Coalesced load: threads read consecutive addresses
2. Reorganize in shared memory
3. Threads access shared memory (fast regardless of pattern)

## Hint 4: Measuring Impact
Use `nvprof --metrics gld_efficiency` to measure coalescing
