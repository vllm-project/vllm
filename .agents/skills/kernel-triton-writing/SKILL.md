---
name: kernel-triton-writing
description: >
  ONLY for OpenAI Triton (@triton.jit) kernel development. NEVER use for
  CUDA C++ kernels, TileIR, or profiling tools (ncu, nsys).
  The user's request must involve Triton explicitly. Covers Triton-specific
  patterns: fused elementwise, reductions (softmax, LayerNorm, RMSNorm),
  tiled GEMM with triton.autotune, and flash attention. Workflow:
  route, design, implement, and validate in vLLM.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
  source: https://github.com/NVIDIA/TensorRT-LLM
  source_commit: 395985c025c8d1cf5aa842bc752b337ba88721b6
---

# Triton Kernel Writing

<!--
Copied from NVIDIA TensorRT-LLM's kernel-triton-writing skill at commit
395985c025c8d1cf5aa842bc752b337ba88721b6. Copyright NVIDIA Corporation;
licensed under Apache-2.0. See ORIGIN.md for the source and local changes.
-->

<!-- markdownlint-disable MD040 MD060 -->

## Principles

### Correctness First

1. Never benchmark before verification passes.
2. Always mask loads and stores for non-divisible shapes.
3. Validate the public wrapper against a PyTorch reference in the nearest
   existing pytest suite.
4. Cover the relevant shapes, dtypes, strides, and boundary conditions with
   explicit tolerances.

### FP16/BF16 Precision Rules (LOW FREEDOM -- follow exactly)

Transcendental functions (`tl.exp`, `tl.log`, `tl.math.erf`, `tl.math.tanh`) require fp32 inputs.

```python
# WRONG -- compilation error or wrong results with fp16/bf16:
result = tl.exp(x)

# CORRECT -- cast to fp32, compute, cast back:
x_fp32 = x.to(tl.float32)
result = tl.exp(x_fp32).to(x.dtype)
```

Rule: any math function beyond basic arithmetic (+, -, *, /) requires fp32 cast in, original dtype cast out.

Additional precision constraints:

- `tl.sigmoid()` is unavailable in some Triton versions. Use `1.0 / (1.0 + tl.exp(-x_fp32))`.
- Always cast back to `x.dtype` before `tl.store` -- mismatches cause "Type mismatch, store Float32 to Float16".
- Unlike PyTorch, Triton does NOT auto-promote fp16/bf16 to fp32 for accumulation. Always use `tl.float32` accumulators for `tl.dot`.
- **TF32 for matmul:** On Ampere+/Hopper, `tl.dot` uses TF32 by default for fp32 inputs (same as `torch.mm`). Do NOT add `input_precision="ieee"` — it is 3-8x slower. TF32 is the correct default. If verification fails due to TF32 precision (~0.01-0.1 abs diff), ensure the PyTorch reference also uses TF32 (plain `torch.mm`, no `allow_tf32=False`).

### CPU-GPU Sync Avoidance (LOW FREEDOM)

Never call `.item()` in kernel wrappers. It forces a CPU-GPU sync (~50-100us per call).

| Pitfall | Fix |
|---------|-----|
| `tensor.item()` for seed | `x.data_ptr() % (2**31)` |
| `torch.randint(...).item()` | Use tensor metadata for pseudo-random seed |
| Allocating output every call | Accept pre-allocated output as parameter |
| Python loops calling kernel | Batch operations |

### C Integer Division Semantics (CRITICAL)

Triton uses **C semantics** (round toward zero) for `//` and `%`, NOT Python semantics (round toward negative infinity). This only matters when operands can be negative.

| Expression | Python | Triton/C |
|------------|--------|----------|
| `-7 // 2` | `-4` | `-3` |
| `-7 % 2` | `1` | `-1` |

**Safe pattern:** Ensure all index/offset values are non-negative. If negative values are possible, use `(idx % BLOCK + BLOCK) % BLOCK`.

See [references/concepts-semantics.md](references/concepts-semantics.md) for full rules and scalar-only exception.

### Kernel Design Mental Model

- **Parallelization axis:** Element-wise kernels parallelize over flattened elements. Row-wise kernels (LayerNorm, softmax) parallelize over rows. Matmul kernels tile in 2D (M, N).
- **Block size:** Power-of-2 only (256, 512, 1024, 2048). Start with 1024 for H100, 512 for V100.
- **Memory coalescing:** Adjacent threads must access adjacent memory addresses. The compiler handles this automatically from block-level pointer arithmetic.
- **Grid:** Use `triton.cdiv(n_elements, BLOCK_SIZE)`. With autotune, grid must be a lambda: `lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`.
- **Decorator order:** `@triton.autotune` (outermost) -> `@triton.heuristics` -> `@triton.jit` (innermost).
- **`reset_to_zero`:** Required for autotune on kernels that accumulate (e.g., matmul output). Without it, later configs see leftover values from earlier trials.

## Workflow

**Fast path:** If the user explicitly requests a Triton kernel (e.g., "Write a Triton kernel for X", "Implement softmax in Triton"), start at **Phase 2**. Only use Phase 0-1 when the request is ambiguous about whether Triton is appropriate.

### Phase 0: Route the Operator (only for ambiguous requests)

Skip this phase if the user explicitly asks for a Triton kernel. Only use when the request is ambiguous (e.g., "optimize this operation").

Triton wins when 2+ operations can share registers instead of writing/reading global memory. Quick rules:

| Pattern | Decision |
|---------|----------|
| Single element-wise op (`relu`, `sigmoid`) | SKIP — PyTorch already optimal |
| Standalone matmul | SKIP — cuBLAS is optimized |
| Standard attention | SKIP — Use FlashAttention |
| Element-wise chain (2+ ops), reduction, matmul + epilogue | USE TRITON |

If SKIP, recommend the alternative and STOP. See [references/operator-routing.md](references/operator-routing.md) for edge cases.

### Phase 1: Analyze the Operator (only for ambiguous requests)

From the user's request, identify: (1) operation type, (2) parallelization strategy, (3) input shapes and dtypes.

### Phase 2: Design the Kernel

Pick the skeleton below that matches your operation. **These skeletons are sufficient for element-wise, reduction, matmul, and fusion kernels — do NOT read reference files for these common patterns.** Only consult `references/` when implementing uncommon patterns (grouped GEMM, TMA, extern functions) or debugging issues.

**Element-wise skeleton** (GELU, dropout, fused ops on flat tensors):

```python
@triton.jit
def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ... compute ...
    tl.store(out_ptr + offsets, result, mask=mask)
```

**Row-wise skeleton** (softmax, LayerNorm, RMSNorm — one program per row):

```python
@triton.jit
def kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    # ... reduce / normalize ...
    tl.store(out_ptr + row_idx * n_cols + col_offsets, result, mask=mask)
```

**Tiled matmul skeleton** (GEMM with 2D tiling, grouped ordering, and autotune):

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    # Grouped ordering for L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_n_blocks
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_m_blocks - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
```

### Phase 3: Write the Kernel

Implement the kernel in the appropriate vLLM module and match nearby public
interfaces and style. The implementation should include:

- `@triton.jit` decorated kernel function
- `@triton.autotune` for production kernels (see [references/api-core.md](references/api-core.md))
- Python wrapper function (descriptive name for external import)

Concise example (fused GELU + dropout):

```python
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_dropout_kernel(
    x_ptr, out_ptr, n_elements, p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    x = (0.5 * x_fp32 * (1.0 + tl.math.erf(x_fp32 * 0.7071067811865476))).to(x.dtype)

    random = tl.rand(seed, offsets)
    x = tl.where(random > p, x / (1.0 - p), 0.0)

    tl.store(out_ptr + offsets, x, mask=mask)


def fused_gelu_dropout_triton(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    seed = (x.data_ptr() % (2**31)) ^ n_elements  # sync-free seed
    fused_gelu_dropout_kernel[grid](x, out, n_elements, p, seed)
    return out
```

For more patterns (SiLU+mul, RMSNorm, linear+GELU, add+LayerNorm), see [references/patterns-fusion.md](references/patterns-fusion.md). For GEMM patterns, see [references/patterns-gemm.md](references/patterns-gemm.md).

### Phase 4: Verify Correctness

Extend the nearest existing pytest suite under `tests/kernels/`. Compare the
public wrapper against a PyTorch reference and use the smallest cases that
cover the intended shapes, dtypes, strides, non-divisible boundaries, and any
in-place behavior. Keep tolerances explicit.

Run the focused test file:

```bash
.venv/bin/python -m pytest {test_path} -v
```

Stop and fix the kernel if correctness fails. Do not use benchmark agreement
as a substitute for a pytest regression test.

**Tolerance guide:**

| Dtype | rtol | atol | Notes |
|-------|------|------|-------|
| float16 | 1e-3 | 1e-3 | |
| bfloat16 | 1e-2 | 1e-2 | |
| float32 | 1e-5 | 1e-5 | Element-wise ops |
| float32 (matmul) | 1e-2 | 1e-1 | TF32 accumulation order differs between Triton tiles and cuBLAS |

### Phase 5: Benchmark Performance (optional)

Only benchmark if the user explicitly requests performance numbers. Skip this phase for correctness-focused requests.

Use `$kernel-microbenchmark` for benchmark design and interpretation. Put
persistent kernel performance work in `benchmarks/kernels/` and follow its
CUPTI timing, cold-L2, CUDA graph, throughput, and reproducibility guidance.

## References (consult only when stuck)

The skeletons and principles above cover element-wise, reduction, matmul, and fusion kernels. **Do NOT read reference files for these common patterns.**

Only consult `references/` when:

- Implementing **uncommon patterns** (grouped GEMM, TMA, persistent matmul, extern functions)
- **Debugging** a compile error or incorrect result not covered by the error table below
- Needing **API details** for an unfamiliar `tl.*` operation

**How to search:** Grep for your keyword across `references/`. Read only the file Grep points to.

| File | When to use |
|---|---|
| `references/api-core.md` | Unfamiliar `triton.autotune` / `triton.Config` options |
| `references/api-language.md` | Unfamiliar `tl.*` operations |
| `references/patterns-gemm.md` | Grouped GEMM, persistent matmul, TMA, MX formats |
| `references/patterns-advanced.md` | Flash attention details, backward passes, libdevice |
| `references/troubleshooting.md` | Debug ops, interpreter mode, env vars |

## Error Handling and Troubleshooting

### Common Errors

| Error / Symptom | Cause | Fix |
|---------|-------|-----|
| "Type mismatch, store Float32 to Float16" | Missing `.to(x.dtype)` before store | Cast fp32 result back |
| `BLOCK_SIZE is not a constexpr` | Block size passed as runtime value | Add `: tl.constexpr` annotation |
| `shape mismatch` in binary op | Tensor shapes don't broadcast | Check with `tl.static_print`; use `[:, None]` / `[None, :]` |
| Large diffs everywhere | Wrong dtype in `tl.load` | Check load dtype matches input |
| Matmul 3-8x slower than expected | `input_precision="ieee"` on `tl.dot` | Remove it; use TF32 default. Ensure the PyTorch reference also uses TF32 |
| Matmul ~0.01-0.1 abs diff vs reference | TF32 vs IEEE mismatch | Use same precision in both kernel and reference (TF32 for both) |
| Diffs at boundaries | Missing mask | Add mask to all load/store ops |
| Random diffs | Race condition | Check atomics and ordering |
| NaN/Inf | Division by zero or fp16 overflow | Guard with epsilon; use `tl.float32` accumulator |
| `grid must be a tuple` | Grid lambda returns int, not tuple | Return `(value,)` with trailing comma |
| `expected constexpr` in `tl.arange` | Non-constexpr argument | Both args of `tl.arange(start, end)` must be constexpr |
| `triton.OutOfResources` | Register/shared memory pressure | Reduce BLOCK_SIZE or `num_stages` |
| Kernel not updating after edit | Stale compilation cache | Move the confirmed Triton cache directory aside and retry |
| Mismatched results vs PyTorch | C integer division semantics | Triton uses truncation; see `references/concepts-semantics.md` |

For extended error table, interpreter mode issues, and environment variables, see [references/troubleshooting.md](references/troubleshooting.md).

### When to Abort

Stop and report failure if:

1. **Not a good fit** -- Pure matmul or complex control flow (Phase 0 should catch this).
2. **Verification fails after 3 attempts** -- Numerical issues too severe to fix.
3. **No speedup** -- Reference is already well-optimized (cuBLAS, cuDNN).
4. **Hardware mismatch** -- Target GPU not available for testing.
