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

# Triton Language API (`triton.language` / `tl`)

## Programming Model

| Function | Signature | Notes |
|---|---|---|
| `program_id` | `program_id(axis)` | Returns ID of current program instance along `axis` (0, 1, or 2) |
| `num_programs` | `num_programs(axis)` | Returns number of program instances along `axis` |
| `tensor` | N-D array type | Block-structured; all ops are implicitly vectorized over the block |
| `tensor_descriptor` | Returned by `make_tensor_descriptor` | Opaque handle; backed by TMA on supported NVIDIA GPUs |

## Creation Operations

| Function | Signature | Notes |
|---|---|---|
| `arange` | `arange(start, end)` | Half-open `[start, end)`, returns 1-D int32 tensor |
| `full` | `full(shape, value, dtype)` | Broadcast scalar `value` to `shape` |
| `zeros` | `zeros(shape, dtype)` | Shorthand for `full(shape, 0, dtype)` |
| `zeros_like` | `zeros_like(x)` | Zeros with same shape/dtype as `x` |
| `cat` | `cat(x, y, can_reorder=False)` | Concatenate along dim 0; `can_reorder` allows compiler flexibility |
| `cast` | `cast(x, dtype, fp_downcast_rounding="rtne")` | Type conversion; rounding: `"rtne"` (default) or `"rtz"` |

## Memory Operations (Pointer-based)

### `tl.load` -- most-used memory op

```python
load(pointer, mask=None, other=None, boundary_check=(),
     padding_option='', cache_modifier='', eviction_policy='', volatile=False)
```

**Key semantics:**

- `mask`: block of `int1`. Where False, returns `other` (default 0). Required for out-of-bounds safety.
- `other`: fallback value where mask is False. Must match dtype.
- `boundary_check`: tuple of dims for block-pointer bounds checking (mutually exclusive with `mask`).
- `padding_option`: `"zero"` or `"nan"` (only with `boundary_check`).
- `cache_modifier`: `""`, `".cg"`, `".cs"`, `".ca"`, `".wb"`, `".wt"`.
- `eviction_policy`: `""`, `"evict_first"`, `"evict_last"`.

```python
# Typical masked load pattern
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n_elements
x = tl.load(ptr + offs, mask=mask, other=0.0)
```

### `tl.store`

```python
store(pointer, value, mask=None, boundary_check=(),
      cache_modifier='', eviction_policy='')
```

Same mask semantics as load. Where mask is False, store is skipped (no side effect).

```python
tl.store(out_ptr + offs, result, mask=mask)
```

## Memory Operations (Block Pointer)

| Function | Signature | Notes |
|---|---|---|
| `make_block_ptr` | `(base, shape, strides, offsets, block_shape, order)` | Structured pointer; `order` controls memory layout (e.g., `(1,0)` for col-major) |
| `advance` | `advance(block_ptr, offsets)` | Returns NEW ptr (no mutation); `offsets` is tuple by dim |

```python
a_ptr = tl.make_block_ptr(a, (M, K), (stride_am, stride_ak), (pid_m * BM, 0), (BM, BK), order=(1, 0))
a_ptr = tl.advance(a_ptr, (0, BK))  # advance K dimension
a = tl.load(a_ptr, boundary_check=(0, 1))
```

## Memory Operations (Tensor Descriptor / TMA)

```python
make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero")
# base must be 16-byte aligned. Supports 2-5D tensors.
# On NVIDIA GPUs with TMA, uses hardware TMA descriptor.
```

| Function | Signature | Notes |
|---|---|---|
| `tensor_descriptor.load` | `.load(offsets, boundary_check=True)` | Load block at `offsets` from descriptor |
| `tensor_descriptor.store` | `.store(offsets, value)` | Store block at `offsets` |

## Linear Algebra

### `tl.dot`

```python
dot(input, other, acc=None, input_precision="tf32", max_num_imprecise_acc=None, out_dtype=float32)
```

- Both operands must be 2-D or 3-D (batched matmul). Inner dims must match (min 16).
- `input` dtype: int8, float8_e5m2, float8_e4m3fn, float16, bfloat16, float32.
- `input_precision`: `"tf32"` (default, NVIDIA), `"tf32x3"`, `"ieee"`.
- `acc`: accumulator tensor; if provided, result is added to it.

### `tl.dot_scaled` (Microscaling / MX formats)

```python
dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
           acc=None, out_dtype=float32)
```

- Formats: `"e2m1"`, `"e4m3"`, `"e5m2"`, `"bf16"`, `"fp16"`.
- Scales are e8m0 (uint8 tensors), shape `[M, K//group_size]`.

## Math Operations

| Function | Signature | Notes |
|---|---|---|
| `abs` | `abs(x)` | Elementwise absolute value |
| `cdiv` | `cdiv(x, div)` | Ceiling division: `(x + div - 1) // div` |
| `ceil` | `ceil(x)` | Ceiling (float) |
| `floor` | `floor(x)` | Floor (float) |
| `exp` | `exp(x)` | Base-e exponential |
| `exp2` | `exp2(x)` | Base-2 exponential |
| `log` | `log(x)` | Natural logarithm |
| `log2` | `log2(x)` | Base-2 logarithm |
| `cos` | `cos(x)` | Cosine |
| `sin` | `sin(x)` | Sine |
| `sqrt` | `sqrt(x)` | Square root |
| `rsqrt` | `rsqrt(x)` | Reciprocal square root: `1/sqrt(x)` |
| `sigmoid` | `sigmoid(x)` | `1 / (1 + exp(-x))` |
| `softmax` | `softmax(x, axis)` | Numerically-stable softmax along `axis` |
| `umulhi` | `umulhi(x, y)` | Upper 32 bits of `x * y` (uint32) |
| `fdiv` | `fdiv(x, y, ieee_rounding=False)` | Floating-point division |
| `fma` | `fma(x, y, z)` | Fused multiply-add: `x * y + z` |
| `clamp` | `clamp(x, min, max)` | Clamp to range `[min, max]` |
| `minimum` | `minimum(x, y)` | Elementwise min (propagates NaN) |
| `maximum` | `maximum(x, y)` | Elementwise max (propagates NaN) |

## Where (Critical for Masking)

```python
where(condition, x, y)
```

Returns elements from `x` where `condition` is True, else from `y`. Both `x` and `y` are broadcast to `condition`'s shape. This is the primary tool for conditional logic in Triton.

```python
# Causal mask in attention
mask = offs_m[:, None] >= offs_n[None, :]
attn = tl.where(mask, attn, float("-inf"))
```

## Reduction Operations

All reductions: `fn(input, axis=None, keep_dims=False)`. When `axis=None`, reduces all dims.

| Function | Signature | Notes |
|---|---|---|
| `max` | `max(input, axis, keep_dims=False)` | Maximum along axis |
| `min` | `min(input, axis, keep_dims=False)` | Minimum along axis |
| `argmax` | `argmax(input, axis)` | Index of max along axis |
| `argmin` | `argmin(input, axis)` | Index of min along axis |
| `sum` | `sum(input, axis, keep_dims=False, dtype=None)` | Sum; int/bool auto-upcast to int32, float to float32 |
| `xor_sum` | `xor_sum(input, axis)` | XOR reduction along axis |
| `reduce` | `reduce(input, axis, combine_fn, keep_dims=False)` | Generic reduction with user-defined `combine_fn(a, b) -> c` |

```python
# Reduction pattern: online softmax
row_max = tl.max(row, axis=1, keep_dims=True)
row = tl.exp(row - row_max)
row_sum = tl.sum(row, axis=1, keep_dims=True)
```

## Scan and Sort Operations

| Function | Signature | Notes |
|---|---|---|
| `associative_scan` | `associative_scan(input, axis, combine_fn, reverse=False)` | Prefix scan with user-defined associative `combine_fn` |
| `cumsum` | `cumsum(input, axis, dtype=None)` | Cumulative sum (specialization of scan) |
| `cumprod` | `cumprod(input, axis, dtype=None)` | Cumulative product |
| `histogram` | `histogram(input, num_bins)` | Counts per bin; input values are bin indices |
| `sort` | `sort(input, axis=-1, descending=False, stable=True)` | Sort along axis |
| `topk` | `topk(input, k, axis=-1, descending=True)` | Top-k values along axis |
| `gather` | `gather(input, indices, axis)` | Gather elements along axis using indices |

## Atomic Operations

All atomics: `atomic_*(pointer, val, mask=None, sem="acq_rel", scope="gpu")`.

- `sem`: `"acquire"`, `"release"`, `"acq_rel"` (default), `"relaxed"`.
- `scope`: `"gpu"` (default), `"cta"` (thread block), `"sys"` (system).

| Function | Signature | Notes |
|---|---|---|
| `atomic_add` | `(ptr, val, mask=None, sem, scope)` | Atomic add; returns old value |
| `atomic_max` | `(ptr, val, mask=None, sem, scope)` | Atomic max; returns old value |
| `atomic_min` | `(ptr, val, mask=None, sem, scope)` | Atomic min; returns old value |
| `atomic_and` | `(ptr, val, mask=None, sem, scope)` | Atomic bitwise AND |
| `atomic_or` | `(ptr, val, mask=None, sem, scope)` | Atomic bitwise OR |
| `atomic_xor` | `(ptr, val, mask=None, sem, scope)` | Atomic bitwise XOR |
| `atomic_xchg` | `(ptr, val, mask=None, sem, scope)` | Atomic exchange; returns old value |
| `atomic_cas` | `(ptr, cmp, val, sem, scope)` | Compare-and-swap: if `*ptr == cmp`, set to `val`; returns old value |

## Random Number Generation (Philox PRNG)

| Function | Signature | Notes |
|---|---|---|
| `randint4x` | `randint4x(seed, offset)` | Returns 4 blocks of int32; fastest for multiple streams |
| `randint` | `randint(seed, offset, n_rounds=6)` | Single block of random int32 |
| `rand` | `rand(seed, offset, n_rounds=6)` | Uniform float32 in `[0, 1)` |
| `randn` | `randn(seed, offset, n_rounds=6)` | Normal distribution (float32) |

`seed`: scalar int32. `offset`: block of int32 (determines which element gets which random value).

## Iterators

| Function | Signature | Notes |
|---|---|---|
| `range` | `range(start, stop, step=1)` | Dynamic loop; bounds can be runtime values |
| `static_range` | `static_range(start, stop, step=1)` | Fully unrolled at compile time; bounds must be `constexpr` |

## Debug Operations

| Function | Signature | Notes |
|---|---|---|
| `static_print` | `static_print(*args)` | Print at **compile time**; same interface as Python `print` |
| `static_assert` | `static_assert(cond, msg="")` | Assert at **compile time** |
| `device_print` | `device_print(prefix, *args)` | Print at **runtime** on device; first arg must be string, rest are scalars/tensors |
| `device_assert` | `device_assert(cond, msg="")` | Assert at **runtime**; requires `TRITON_DEBUG=1` env var |

## Compiler Hints

| Function | Signature | Notes |
|---|---|---|
| `assume` | `assume(cond)` | Hint to backend for address calculation optimization |
| `max_contiguous` | `max_contiguous(input, values)` | Declare max contiguous extent per dim; enables coalesced access |
| `max_constancy` | `max_constancy(input, values)` | Declare max constant extent per dim |
| `multiple_of` | `multiple_of(input, values)` | Declare that values are multiples of given constants |
| `debug_barrier` | `debug_barrier()` | Thread barrier (debugging only; not for correctness) |

## Shape Manipulation

| Function | Signature | Notes |
|---|---|---|
| `broadcast` | `broadcast(x, y)` | Broadcast `x` and `y` to compatible shape (returns both) |
| `broadcast_to` | `broadcast_to(x, shape)` | Broadcast `x` to explicit `shape` |
| `expand_dims` | `expand_dims(x, axis)` | Insert length-1 dim at `axis` |
| `reshape` | `reshape(x, shape)` | Reshape (total elements must match) |
| `view` | `view(x, shape)` | Like reshape; bitcast semantics |
| `trans` | `trans(x, *dims)` | Transpose; default swaps last two dims |
| `permute` | `permute(x, *dims)` | Reorder dims; `permute(x, 2, 1, 0)` or `permute(x, (2,1,0))` |
| `ravel` | `ravel(x)` | Flatten to 1-D |
| `split` | `split(x)` | Split first dim into separate tensors |
| `join` | `join(x, y)` | Concatenate along a new innermost dim |
| `interleave` | `interleave(x, y)` | Interleave elements from `x` and `y` |

## Inline Assembly

```python
inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)
```

- `asm`: PTX/ASM string with `$0`, `$1`, ... placeholders.
- `constraints`: register constraint string (e.g., `"=r,r"` for one int output, one int input).
- `args`: list of input tensors.
- `dtype`: output dtype (or tuple for multi-output).
- `is_pure`: True if no side effects (enables CSE).
- `pack`: number of elements per register (typically 1).
