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

# Advanced Triton Patterns

Source tutorials:

- [05-layer-norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [06-fused-attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [07-extern-functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)

## Layer Normalization

Layer norm normalizes across the hidden dimension: `y = (x - mean) / sqrt(var + eps) * w + b`.
Each program instance processes one row of the input (one token). The hidden dimension
is tiled into blocks so arbitrary sizes are supported.

### Forward Kernel (Complete)

```python
@triton.jit
def _layer_norm_fwd_fused(
    X,       # input pointer,  shape (M, N)
    Y,       # output pointer, shape (M, N)
    W,       # weight pointer, shape (N,)
    B,       # bias pointer,   shape (N,)
    Mean,    # mean pointer,   shape (M,) — written for backward
    Rstd,    # rstd pointer,   shape (M,) — written for backward
    stride,  # row stride of X and Y
    N,       # number of columns (hidden size)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # --- Compute mean ---
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # --- Compute variance ---
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store mean and rstd for backward
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # --- Normalize and apply affine transform ---
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

**Key pattern:** Two-pass reduction (mean then variance) using block-wise tiling.
Each loop iteration processes `BLOCK_SIZE` elements with masking for the tail.
Mean and rstd are saved for the backward pass.

### Backward Kernel: Atomic Lock Pattern

The backward pass computes `dw` and `db` which require reducing across all rows
(all programs contribute). Triton uses a spin-lock pattern with `atomic_cas` for
mutual exclusion:

```python
@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB,               # output accumulators, shape (N,)
    DWEIGHT, DBIAS,       # partial sums buffer, shape (GROUP_SIZE_M, N)
    Lock,                 # lock array, shape (1,) — int32
    ...
    GROUP_SIZE_M: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    # Each group of rows accumulates partials then atomically adds to DW/DB

    # --- Compute partial dw, db for this row group ---
    # (loop over assigned rows, accumulate _dw and _db)

    # --- Acquire lock ---
    lock_id = tl.program_id(1)  # column block index
    Lock += lock_id
    Count = Lock + tl.num_programs(1)  # second half stores count

    while tl.atomic_cas(Lock, 0, 1) == 1:  # spin until we get 0->1
        pass
    count = tl.load(Count)  # how many groups have accumulated so far

    if count == 0:
        # First group: just store
        tl.store(DWEIGHT + cols, _dw, mask=mask)
        tl.store(DBIAS + cols, _db, mask=mask)
    else:
        # Subsequent groups: accumulate
        _dw += tl.load(DWEIGHT + cols, mask=mask)
        _db += tl.load(DBIAS + cols, mask=mask)
        tl.store(DWEIGHT + cols, _dw, mask=mask)
        tl.store(DBIAS + cols, _db, mask=mask)

    if count == GROUP_SIZE_M - 1:
        # Last group: write final result
        tl.store(DW + cols, _dw, mask=mask)
        tl.store(DB + cols, _db, mask=mask)

    # --- Release lock and increment count ---
    tl.atomic_xchg(Lock, 0)      # release: set lock back to 0
    tl.store(Count, count + 1)    # must store AFTER release for correctness
    tl.debug_barrier()            # ensure memory operations are visible
```

**Gotchas:**

- `atomic_cas(Lock, 0, 1)` returns the old value; spin while it returns 1 (already held).
- `atomic_xchg(Lock, 0)` unconditionally sets to 0 (release). Do NOT use `atomic_cas` for release.
- The count update (`tl.store(Count, count + 1)`) must happen after the lock is released.
- `tl.debug_barrier()` forces memory ordering visibility across programs.
- Lock array must be zero-initialized before each backward call.
- This pattern is needed because Triton has no native cross-program reduction for non-atomic dtypes.

## Fused Attention

Implements Flash Attention v2: fused Q*K^T softmax and V accumulation in a single
kernel, avoiding materializing the full N x N attention matrix.

### Online Softmax Algorithm

The key insight is computing softmax in a single streaming pass using running
statistics. For each block of K/V columns processed:

```
# For each new block j of keys:
qk = Q_block @ K_block_j^T           # [BLOCK_M, BLOCK_N]
m_ij = max(qk, axis=1)               # new block max
m_i_new = max(m_i, m_ij)             # update running max
alpha = exp(m_i - m_i_new)           # correction factor for old accumulators
p = exp(qk - m_i_new[:, None])       # stable softmax numerator
l_i = alpha * l_i + sum(p, axis=1)   # update running denominator
acc = alpha[:, None] * acc + p @ V_j  # rescale old acc + new contribution
m_i = m_i_new                         # commit new max
# After all blocks:
acc = acc / l_i[:, None]              # final normalization
```

**Why this works:** Each time a new block raises the running max, all previous
accumulations are rescaled by `exp(old_max - new_max)`, maintaining numerical
equivalence to the two-pass softmax.

### Multi-Stage Processing (Causal Masking)

The STAGE parameter controls masking behavior in the inner loop:

| STAGE | Behavior | When used |
|-------|----------|-----------|
| 1 | Off-band: skip blocks entirely below diagonal | Causal, early blocks |
| 3 | On-band: apply causal mask within block | Causal, diagonal blocks |
| 2 | No masking | Non-causal attention |

```python
# Causal masking within a block (STAGE == 3):
if STAGE == 3:
    # Current query rows: [start_m, start_m + BLOCK_M)
    # Current key cols:   [start_n, start_n + BLOCK_N)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal_mask, qk, float("-inf"))
```

**Gotcha:** When `STAGE == 1`, blocks where all keys are below the diagonal are
skipped entirely (the inner loop `start_n` begins past the diagonal). This is a
major performance win for causal attention on long sequences.

### Kernel Structure Skeleton

```python
@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale,
    M,          # log-sum-exp for backward, shape (batch, nheads, seqlen)
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q strides
    # ... K, V, Out strides ...
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,    # query block size (e.g. 128)
    BLOCK_N: tl.constexpr,    # key block size (e.g. 64)
    HEAD_DIM: tl.constexpr,   # head dimension (e.g. 64)
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)     # which query block
    off_hz = tl.program_id(1)      # batch * head index

    # Initialize pointers for Q[start_m], K, V
    # Load Q block into registers (stays resident)
    q = tl.load(Q_block_ptr)       # [BLOCK_M, HEAD_DIM]

    # Accumulator in float32
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

    # --- Inner loop over K/V blocks ---
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)   # [BLOCK_N, HEAD_DIM]
        v = tl.load(V_block_ptr)   # [BLOCK_N, HEAD_DIM]

        qk = tl.dot(q, tl.trans(k)) * sm_scale  # [BLOCK_M, BLOCK_N]

        # Apply causal mask if STAGE == 3
        # Online softmax update (m_i, l_i, acc) as shown above

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Final normalization
    acc = acc / l_i[:, None]
    # Store lse = m_i + log(l_i) for backward
    tl.store(Out_block_ptr, acc.to(Out.type.element_ty))
```

### TensorDescriptor (Hopper+)

On Hopper/Blackwell, `TensorDescriptor` enables TMA (Tensor Memory Accelerator):

```python
desc_q = TensorDescriptor(Q_ptr, shape=[N_CTX, HEAD_DIM],
                          strides=[stride_qm, stride_qk],
                          block_shape=[BLOCK_M, HEAD_DIM])
q = desc_q.load([start_m * BLOCK_M, 0])  # replaces pointer arithmetic
```

### Warp Specialization and Performance

On Blackwell (sm_100+), warp specialization lets different warp groups play
producer/consumer roles, overlapping loads with compute via `tl.async_task`.
Flash Attention in Triton reaches ~165 TFLOPS (fp16, H100). Causal masking
with the STAGE optimization roughly halves unnecessary work.

## External Functions

Triton can call functions from external device libraries (libdevice for CUDA,
ROCm device libs for HIP) for math operations not built into the language.

### Basic Usage

```python
@triton.jit
def asin_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    # Call libdevice asin — dispatches based on dtype (fp32 or fp64)
    y = tl.extra.cuda.libdevice.asin(x)
    tl.store(y_ptr + offset, y, mask=mask)
```

Type dispatch is automatic: `libdevice.asin` calls `__nv_asinf` for float32
and `__nv_asin` for float64 under the hood.

### Custom Library Paths

Pass external libraries explicitly via `extern_libs` at compile time:

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
asin_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024,
                  extern_libs={"libdevice": "/path/to/libdevice.10.bc"})
```

### Backend Detection

| Backend | Library file | Namespace |
|---------|-------------|-----------|
| CUDA    | `libdevice.10.bc` | `tl.extra.cuda.libdevice.*` |
| HIP     | `ocml.bc` / `ockl.bc` | Functions mapped through HIP backend |

Common functions: `asin`, `acos`, `atan`, `exp`, `log`, `pow`, `sqrt`, `rsqrt`,
`fma`, `cbrt`, `erf`, `erfc`, `ceil`, `floor`, `round`.

**Gotcha:** `extern_libs` must point to the `.bc` bitcode file (typically
`/usr/local/cuda/nvvm/libdevice/libdevice.10.bc`). Missing file = compile-time linker error.
