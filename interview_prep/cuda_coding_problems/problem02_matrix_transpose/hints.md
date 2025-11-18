# Hints: Matrix Transpose

## Hint 1: Why Naive Transpose is Slow
```cuda
// Naive: reads are coalesced, but writes are strided!
output[x * rows + y] = input[y * cols + x];
// Writing to column-major order causes poor performance
```

## Hint 2: Tiled Approach
Use 32×32 shared memory tiles:
1. Load tile from input (coalesced)
2. Transpose in shared memory
3. Write to output (now also coalesced)

## Hint 3: Bank Conflict Problem
```cuda
__shared__ float tile[32][32];  // BAD: causes bank conflicts
__shared__ float tile[32][33];  // GOOD: padding eliminates conflicts
```

## Hint 4: Key Indexing
```cuda
// Load: blockIdx.x * TILE + threadIdx.x
// Store: blockIdx.y * TILE + threadIdx.x (swapped!)
output[y * rows + x] = tile[threadIdx.x][threadIdx.y];  // Note swap
```

## Hint 5: Boundary Handling
Always check: `if (x < cols && y < rows)` before accessing memory.
