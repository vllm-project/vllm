# Hints: Attention Scores

## Hint 1: Matrix Multiply Pattern
Q @ K^T means: for each (i,j), compute dot product of Q[i,:] and K[j,:]

## Hint 2: Tiling Strategy
- Tile Q and K in shared memory
- Each tile computes partial dot products
- Accumulate across tiles

## Hint 3: K^T Access
When loading K, swap indices to get transpose effect:
```cuda
K_tile[threadIdx.y][threadIdx.x] = K[k_row * d_k + k_col];
```

## Hint 4: Scaling
Use `rsqrtf(d_k)` for faster 1/sqrt(d_k) computation
