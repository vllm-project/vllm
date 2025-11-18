# Problem 4: Parallel Prefix Sum (Scan)

**Difficulty:** Easy
**Estimated Time:** 35-45 minutes
**Tags:** Parallel Algorithms, Scan, Work-Efficient

## Problem Statement

Implement an efficient parallel prefix sum (scan) algorithm in CUDA. Given an array, compute the cumulative sum at each position. Support both inclusive and exclusive scan.

## Requirements

- Inclusive scan: `output[i] = sum(input[0..i])`
- Exclusive scan: `output[i] = sum(input[0..i-1])`, output[0] = 0
- Use work-efficient algorithm (O(n) work, O(log n) depth)
- Handle arrays larger than block size
- Use shared memory for intra-block scan

## Function Signature

```cuda
__global__ void scanKernel(float* input, float* output, float* blockSums, int n, bool inclusive);
void prefixSum(float* h_input, float* h_output, int n, bool inclusive);
```

## Examples

### Inclusive Scan
```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [3, 4, 11, 11, 15, 16, 22, 25]
```

### Exclusive Scan
```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [0, 3, 4, 11, 11, 15, 16, 22]
```

## Algorithm: Blelloch Scan

**Up-sweep (reduce) phase:**
```
[3, 1, 7, 0, 4, 1, 6, 3]
[_, 4, _, 7, _, 5, _, 9]  // pairs
[_, _, _, 11, _, _, _, 14] // fours
[_, _, _, _, _, _, _, 25]  // total
```

**Down-sweep (distribute) phase:**
Build prefix sums from partial sums.

## Success Criteria

- ✅ Correct inclusive and exclusive scans
- ✅ Work-efficient (Blelloch algorithm)
- ✅ Handles arbitrary array sizes
- ✅ Uses shared memory efficiently
- ✅ Minimal bank conflicts
