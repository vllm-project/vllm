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

# Triton Concepts and Semantics

## Programming Model — Block-Based Execution

Triton programs operate on **blocks** (tiles) of data, not individual scalar threads.
Each kernel instance (called a "program") processes an entire block of elements at once.

| Concept | CUDA | Triton |
|---------|------|--------|
| Execution unit | Single scalar thread | Program operating on a block |
| Memory coalescing | Manual (stride patterns) | Automatic (compiler) |
| Shared memory | Manual (`__shared__`, sync) | Automatic (compiler) |
| Vectorization | Manual (float4, etc.) | Automatic (compiler) |
| Tensor core usage | Manual (wmma/mma) | Automatic (compiler) |
| Thread synchronization | Manual (`__syncthreads`) | Not needed |

### Launch Grid and Program IDs

```python
@triton.jit
def kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program instance gets a unique ID along each grid axis
    pid = tl.program_id(axis=0)  # which block of data this program handles
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    tl.store(Y_ptr + offsets, x * 2, mask=mask)

# Launch with a 1D grid: one program per block of data
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
kernel[grid](x_ptr, y_ptr, N, BLOCK_SIZE=1024)
```

### Key Takeaway

The programmer thinks in blocks. The compiler decides how to map blocks to warps,
how to stage data through shared memory, and when to use tensor cores. This is the
core design tradeoff: less control, but far less boilerplate and fewer correctness bugs.

---

## Type Promotion Rules

Triton applies automatic type promotion for binary ops and `tl.where` (last two args).

### Promotion Hierarchy

```
{bool} < {int8, int16, int32, int64, uint8, uint16, uint32, uint64} < {fp8, fp16, bf16, fp32, fp64}
  ^               ^  (integral types)                                       ^  (floating types)
  kind 0                  kind 1                                           kind 2
```

### Rules Applied in Order

| Priority | Rule | Example |
|----------|------|---------|
| 1 | **Cross-kind**: lower kind promotes to higher kind's dtype | `(int32, bf16)` -> `bf16` |
| 2 | **Same-kind widening**: narrower promotes to wider | `(fp16, fp32)` -> `fp32` |
| 3 | **Same-width float tie**: bf16 and fp16 both promote to fp16 | `(fp16, bf16)` -> `fp16` |
| 4 | **Same-width sign tie**: promote to unsigned | `(int32, uint32)` -> `uint32` |

### Scalar-Tensor Interaction

When a Python scalar interacts with a Triton tensor:

| Scalar Type | Tensor Type | Result |
|-------------|-------------|--------|
| Python `int` | Any int tensor | Tensor's dtype (no widening) |
| Python `int` | Any float tensor | Tensor's dtype |
| Python `float` | Any float tensor | Tensor's dtype |
| Python `float` | Any int tensor | `fp64` (float is higher kind) |

### Gotchas: Type Promotion

| Gotcha | Detail |
|--------|--------|
| `int32 + bf16` -> `bf16` | Integer silently truncated to bf16 precision (only ~3 decimal digits) |
| `fp16 + bf16` -> `fp16` | bf16 promotes to fp16, NOT fp32. May lose bf16 range |
| `int8 + uint8` -> `uint8` | Signed values become unsigned, wrapping negative values |
| Cross-kind hides widening | `(int64, fp16)` -> `fp16`, losing 64-bit integer precision |
| No implicit fp32 promotion | Unlike PyTorch, Triton does NOT auto-promote fp16/bf16 to fp32 for accumulation |

---

## Broadcasting Rules

Triton broadcasting follows NumPy conventions with one key constraint:
tensors are at most 2D in practice (block pointers may extend this).

### Rules

1. **Left-pad with ones**: If tensors have different numbers of dimensions, the
   shorter shape is padded on the left with 1s.
2. **Dimension-1 expansion**: Dimensions of size 1 are stretched to match the
   corresponding dimension of the other tensor.
3. **Incompatible = error**: If dimensions differ and neither is 1, it is a compile error.

### Example: Row-Column Broadcast

```python
# Create a row vector (1, N) and column vector (M, 1)
row = tl.arange(0, N)[None, :]     # shape: (1, N)
col = tl.arange(0, M)[:, None]     # shape: (M, 1)

# Broadcast produces (M, N) — outer product pattern
result = row + col                   # shape: (M, N)
```

### Common Broadcasting Patterns

| Pattern | Shape A | Shape B | Result Shape | Use Case |
|---------|---------|---------|-------------|----------|
| Row + Col | `(1, N)` | `(M, 1)` | `(M, N)` | 2D index grids, outer products |
| Scalar + Block | `()` | `(M, N)` | `(M, N)` | Add bias, scale |
| Row mask | `(1, N)` | `(M, N)` | `(M, N)` | Column-wise masking |

### Gotcha: Broadcasting

| Gotcha | Detail |
|--------|--------|
| No implicit unsqueeze | You must explicitly reshape with `[:, None]` or `[None, :]` |
| 1D + 1D does NOT broadcast | Two 1D tensors of different length are an error, not broadcast |
| Mask must broadcast to data | `tl.load(ptr, mask=mask)` — mask shape must broadcast to ptr block shape |

---

## Integer Division and Modulus — C Semantics

**CRITICAL**: Triton uses **C semantics** (round toward zero), NOT Python semantics
(round toward negative infinity). This is the most common source of subtle bugs
when porting Python logic to Triton kernels.

### Comparison Table

| Expression | Python Result | Triton Result | Why |
|------------|--------------|---------------|-----|
| `-7 // 2` | `-4` | `-3` | Python: floor division. Triton/C: truncation toward zero |
| `-7 % 2` | `1` | `-1` | Follows from division: `a == (a // b) * b + (a % b)` |
| `7 // -2` | `-4` | `-3` | Same: truncation vs floor |
| `7 % -2` | `-1` | `1` | Remainder keeps dividend sign in C |
| `-7 // -2` | `3` | `3` | Both agree when signs match (positive quotient) |

### The Identity

Both C and Python satisfy: `a == (a // b) * b + (a % b)`

But they disagree on which direction to round the quotient, which changes the remainder.

### Exception: Scalar-Only Computations

When **all inputs are Python scalars** (not Triton tensors), division and modulus
follow **Python semantics**. This only applies to compile-time constant folding.

```python
@triton.jit
def kernel(X_ptr, N, BLOCK: tl.constexpr):
    # Python semantics — both are Python scalars at compile time
    blocks_per_row = (-7) // 2    # = -4 (Python floor division)

    # C semantics — pid is a Triton value
    pid = tl.program_id(0)
    row = pid // N                 # truncation toward zero
    col = pid % N                  # C remainder
```

### Gotcha: Safe Patterns for Negative Values

| Unsafe Pattern | Problem | Safe Alternative |
|----------------|---------|------------------|
| `(-offset) // stride` | C truncation gives wrong block | `-(offset // stride)` or use unsigned |
| `idx % BLOCK` for negative idx | Negative remainder | Ensure idx is non-negative, or add `+ BLOCK) % BLOCK` |
| Porting Python `divmod` logic | Both `//` and `%` differ | Rewrite with explicit floor: `q = (a - (a % b + b) % b) // b` |

### When It Matters

This only causes bugs when **operands can be negative**. If all values are
non-negative (which is common for pointer offsets and indices), C and Python
semantics agree. Guard against negative values explicitly when in doubt.
