<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileComment: Copied from NVIDIA TensorRT-LLM at commit 395985c025c8d1cf5aa842bc752b337ba88721b6.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- markdownlint-disable MD040 MD060 -->

# Triton GEMM Patterns

Reusable matrix multiplication patterns from Triton tutorials 03, 08, 09, 10.

## Matrix Multiplication

Block-tiled GEMM with L2 cache optimization. The workhorse pattern for dense matmul.

### Autotune Configs

| Config | BLOCK_M | BLOCK_N | BLOCK_K | num_stages | num_warps |
|--------|---------|---------|---------|------------|-----------|
| 1      | 128     | 256     | 64      | 3          | 8         |
| 2      | 64      | 256     | 32      | 4          | 4         |
| 3      | 128     | 128     | 32      | 4          | 4         |
| 4      | 128     | 64      | 32      | 4          | 4         |
| 5      | 64      | 128     | 32      | 4          | 4         |
| 6      | 128     | 32      | 32      | 4          | 4         |
| 7      | 64      | 32      | 32      | 5          | 2         |
| 8      | 32      | 64      | 32      | 5          | 2         |

All configs use `GROUP_SIZE_M=8`. `key=["M", "N", "K"]` triggers re-autotuning on shape change.

### Complete Kernel

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32,  "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """C = A @ B. A is (M,K), B is (K,N), C is (M,N)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # L2 cache optimization: super-grouping — nearby pids share B columns
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Multi-dimensional pointer arithmetic via broadcasting
    # 1D offset vectors + strides -> 2D block of pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate along K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # Store with boundary masking
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### Launch Wrapper

```python
def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c
```

### Key Patterns

- **Pointer broadcasting:** `offs_row[:, None] * stride_row + offs_col[None, :] * stride_col` creates 2D pointer block from 1D offsets.
- **tl.dot(a, b, acc):** Accumulates `a @ b` into `acc`. Always use float32 accumulator.
- **Super-grouping:** `GROUP_SIZE_M` controls how many M-tiles share N-tiles, improving L2 hit rate on B. Typically 8.
- **Boundary masking:** `% M`/`% N` wraps OOB offsets to valid addresses (loads still masked). K-dim uses explicit `mask=offs_k < remaining`.

## Grouped GEMM

Batched independent matmuls in a single persistent kernel. Use case: mixture-of-experts (MoE).

### Core Pattern

```python
@triton.jit
def grouped_matmul_kernel(
    a_ptrs, b_ptrs, c_ptrs,      # device arrays of per-group pointers
    m_sizes, n_sizes, k_sizes,    # per-group dimensions
    lds_a, lds_b, lds_c,         # per-group leading dimensions
    group_offsets,                 # cumulative tile count per group
    num_tiles,
    NUM_SM: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    # Persistent: each SM strides across all tiles
    for tile_idx in tl.range(tile_idx, num_tiles, NUM_SM, num_stages=0):
        # Binary-search group_offsets to find which group owns this tile
        # Compute (pid_m, pid_n) within that group
        # Standard matmul accumulation loop for group's (M,K) x (K,N)
        ...

# Launch: one program per SM
NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
grouped_matmul_kernel[(NUM_SM,)](
    a_ptrs, b_ptrs, c_ptrs, m_sizes, n_sizes, k_sizes,
    lds_a, lds_b, lds_c, group_offsets, total_tiles,
    NUM_SM=NUM_SM, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
)
```

### TMA Variant (Hopper+)

```python
# Device-side TMA descriptors — shape varies per group
desc_a = tl.make_tensor_descriptor(
    a_group_ptr, shape=[M_g, K_g], strides=[lda, 1],
    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
)
desc_b = tl.make_tensor_descriptor(
    b_group_ptr, shape=[K_g, N_g], strides=[ldb, 1],
    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
)
a = desc_a.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
b = desc_b.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
```

### Key Patterns

- **Persistent kernel:** Grid = NUM_SM. Each program loops via `tl.range(pid, total, NUM_SM)`.
- **Device-side scheduling:** Binary search on cumulative tile offsets maps flat tile_id to (group, tile_m, tile_n).
- **`tl.make_tensor_descriptor`:** Creates TMA descriptors on-device (needed because shape changes per group).
- **`num_stages=0`:** Outer loop has no pipelining; inner K loop is pipelined.

## Persistent Matmul

TMA descriptors and warp specialization for Hopper/Blackwell. Three progressive variants.

### Variant 1: Persistent with Pointer Arithmetic

Same as basic GEMM but with `tl.range(start_pid, num_tiles, NUM_SM)` outer loop
and grid = `(NUM_SM,)`. See Grouped GEMM pattern above for the persistent loop structure.

### Variant 2: TMA Descriptors

```python
# Host-side: create TMA descriptors before launch
from triton.tools.experimental_descriptor import create_2d_tma_descriptor
desc_a = create_2d_tma_descriptor(a_ptr, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K, a.element_size())
desc_b = create_2d_tma_descriptor(b_ptr, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N, b.element_size())

# Kernel: load via hardware TMA unit (no manual pointer math)
@triton.jit
def matmul_kernel_tma(desc_a, desc_b, c_ptr, M, N, K, ...):
    # Inside K-loop:
    a = tl._experimental_descriptor_load(
        desc_a, [pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K],
        [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
    b = tl._experimental_descriptor_load(
        desc_b, [k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N],
        [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16)
    accumulator = tl.dot(a, b, accumulator)
```

### Variant 3: Warp Specialization

```python
# Warps split into producers (TMA loads) and consumers (tl.dot compute).
# Compiler manages producer/consumer synchronization automatically.
matmul_kernel_warp_spec[(NUM_SM,)](
    desc_a, desc_b, c, M, N, K, ...,
    BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
    num_stages=4, num_warps=8,
    num_consumer_groups=2,       # warp groups for compute
    num_buffers_warp_spec=4,     # pipeline depth for producer/consumer overlap
)
```

### FP8 Support (compute capability >= 9.0)

```python
a = tl.load(a_ptrs, mask=..., other=0.0).to(tl.float8e5m2)  # or tl.float8e4m3fn
b = tl.load(b_ptrs, mask=..., other=0.0).to(tl.float8e5m2)
accumulator = tl.dot(a, b, accumulator)  # acc stays float32
```

### Key Patterns

- **tl.range(start, end, step):** Persistent loop with SM-count stride.
- **TMA descriptors:** Host-side `create_2d_tma_descriptor` or device-side `tl.make_tensor_descriptor`. Offloads address gen to hardware.
- **Warp specialization:** `num_consumer_groups` + `num_buffers_warp_spec` split warps into memory producers and compute consumers.
- **Epilogue subtiling:** Slice accumulator along N for the store phase to cut register pressure.

## Block-Scaled Matmul

Per-block scale factors for microscaling (MX) formats.
Requires 5th-gen Tensor Cores (compute capability >= 10.0, Blackwell+).

### Supported Formats

| Format | Element Type | Scale Block Size | Platform |
|--------|-------------|------------------|----------|
| mxfp8  | float8 (e5m2 / e4m3) | 32 elements | NVIDIA + AMD |
| mxfp4  | float4 (e2m1) | 32 elements | NVIDIA + AMD |
| nvfp4  | float4 (e2m1) | 16 elements | NVIDIA only |

### Kernel with tl.dot_scaled

```python
@triton.jit
def matmul_kernel_block_scaled(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k, stride_b_scale_n, stride_b_scale_k,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # ... super-grouping (same as basic GEMM) ...

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=..., other=0.0)
        b = tl.load(b_ptrs, mask=..., other=0.0)
        a_scale = tl.load(a_scale_ptrs)   # [BLOCK_M, BLOCK_K // 32]
        b_scale = tl.load(b_scale_ptrs)   # [BLOCK_N, BLOCK_K // 32]
        # Hardware-accelerated scaled dot product
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", accumulator)
        # advance pointers ...
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)
```

### Key Patterns

- **tl.dot_scaled(a, a_scale, a_fmt, b, b_scale, b_fmt, acc):** Single instruction applying per-block scales during matmul. Replaces manual dequant-then-multiply.
- **Format strings:** `"e4m3"`, `"e5m2"`, `"e2m1"` passed to `tl.dot_scaled`.
- **Preshuffling:** Always preprocess scales into vendor-specific layout before kernel launch.
- **Hardware:** NVIDIA CC >= 10.0 (Blackwell, PTX 8.7+). AMD CDNA3+ (MI300X).
