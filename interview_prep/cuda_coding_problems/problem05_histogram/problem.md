# Problem 5: Histogram with Atomic Operations

**Difficulty:** Easy
**Estimated Time:** 30-40 minutes
**Tags:** Atomic Operations, Memory Contention, Privatization

## Problem Statement

Implement an efficient histogram kernel in CUDA. Given an array of values and bin count, compute the frequency of values in each bin using atomic operations.

## Requirements

- Count frequencies of values into fixed number of bins
- Use atomic operations correctly
- Optimize for reduced memory contention
- Handle arbitrary input sizes
- Optional: Use privatization/shared memory optimization

## Function Signature

```cuda
__global__ void histogramKernel(int* input, int* histogram, int n, int numBins);
void computeHistogram(int* h_input, int* h_histogram, int n, int numBins);
```

## Input/Output

**Input:**
- `input`: Array of integers (0 ≤ input[i] < numBins)
- `n`: Number of elements
- `numBins`: Number of histogram bins

**Output:**
- `histogram`: Array of counts (length numBins)

## Example

```
Input: [0, 1, 2, 1, 0, 3, 2, 1, 0, 2]
numBins: 4
Output: [3, 3, 3, 1]  // bin 0: 3 counts, bin 1: 3 counts, etc.
```

## Key Concepts

1. **Global Memory Atomics:** Simple but slow due to contention
2. **Shared Memory Privatization:** Each block has private histogram
3. **Warp Aggregation:** Reduce atomic operations
4. **Bank Conflicts:** Avoid in shared memory histograms

## Success Criteria

- ✅ Correct histogram for all test cases
- ✅ Uses atomic operations correctly
- ✅ Handles input sizes >> bin count
- ✅ Bonus: Privatization optimization
- ✅ No race conditions
