# Hints: Dynamic Shared Memory

## Hint 1: Extern Declaration
```cuda
extern __shared__ float tile[];  // Size determined at launch
```

## Hint 2: Launch with Size
```cuda
size_t sharedMem = tileSize * (tileSize + 1) * sizeof(float);
kernel<<<grid, block, sharedMem>>>(...);
```

## Hint 3: 1D to 2D Indexing
Treat 1D array as 2D with padding:
```cuda
tile[row * (tileSize + 1) + col]
```

## Hint 4: Why Padding?
+1 in stride eliminates bank conflicts
