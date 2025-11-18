# Hints: Batched MatMul

## Hint 1: 3D Grid
Use blockIdx.z for batch dimension:
```cuda
dim3 grid(N_blocks, M_blocks, batch_size);
int batch = blockIdx.z;
```

## Hint 2: Memory Offsets
Each batch's data is contiguous:
```cuda
int A_offset = batch * M * K;
int B_offset = batch * K * N;
int C_offset = batch * M * N;
```

## Hint 3: Tiling
Same tiling strategy as regular matmul, but per-batch
