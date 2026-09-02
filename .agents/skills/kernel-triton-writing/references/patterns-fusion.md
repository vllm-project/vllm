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

# Deep Learning Fusion Patterns

Ready-to-use Triton kernel patterns for common DL operator fusions.
Each pattern includes autotune configs, kernel, wrapper, and expected speedups.

For foundational patterns (vector add, softmax, dropout), see `patterns-basic.md`.
For LayerNorm with backward, fused attention, and extern functions, see `patterns-advanced.md`.

---

## GELU + Dropout

**Use when:** Transformer FFN layers with dropout.
**Expected speedup:** 1.8-2.2x vs separate ops.

```python
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

    # GELU (exact): cast to fp32 for erf, then cast back
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1.0 + tl.math.erf(x_fp32 * 0.7071067811865476))
    x = x_gelu.to(x.dtype)

    # Dropout
    random = tl.rand(seed, offsets)
    x = tl.where(random > p, x / (1 - p), 0.0)

    tl.store(out_ptr + offsets, x, mask=mask)


def fused_gelu_dropout(x: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
    if not training or p == 0.0:
        return torch.nn.functional.gelu(x)
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    seed = (x.data_ptr() % (2**31)) ^ n_elements
    fused_gelu_dropout_kernel[grid](x, out, n_elements, p, seed)
    return out
```

---

## SiLU + Multiply (SwiGLU)

**Use when:** LLaMA-style FFN with SwiGLU activation.
**Expected speedup:** 1.5-2x vs `F.silu(gate) * x`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_silu_mul_kernel(
    x_ptr, gate_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)

    # SiLU(gate) * x = gate * sigmoid(gate) * x
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * x

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    assert x.shape == gate.shape
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_silu_mul_kernel[grid](x, gate, out, n_elements)
    return out
```

---

## Residual Add + Activation

**Use when:** Adding residual connection with activation.
**Expected speedup:** 1.4-1.8x vs `F.gelu(x + residual)`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_residual_gelu_kernel(
    x_ptr, residual_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    x = x + residual

    # GELU (exact)
    x_fp32 = x.to(tl.float32)
    x = (0.5 * x_fp32 * (1.0 + tl.math.erf(x_fp32 * 0.7071067811865476))).to(x.dtype)

    tl.store(out_ptr + offsets, x, mask=mask)


def fused_residual_gelu(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_residual_gelu_kernel[grid](x, residual, out, n_elements)
    return out
```

---

## RMSNorm

**Use when:** LLaMA-style normalization (no mean subtraction).
**Expected speedup:** 1.4-2x vs naive PyTorch RMSNorm.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr, out_ptr, weight_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Compute RMS
    x_sq = x * x
    rms = tl.sqrt(tl.sum(x_sq, axis=0) / n_cols + eps)

    # Normalize and scale
    x_norm = x / rms
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    out = x_norm * weight

    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    grid = (n_rows,)
    rmsnorm_kernel[grid](x, out, weight, n_rows, n_cols, eps)
    return out.view(shape)
```

---

## Linear + GELU (Matmul + Epilogue)

**Use when:** Transformer FFN first linear with activation.
**Expected speedup:** 1.3-1.6x vs `F.gelu(F.linear(x, weight, bias))`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = weight_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    # Add bias + fused GELU epilogue
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and weight.is_contiguous()
    M, K = x.shape
    K2, N = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_gelu_kernel[grid](x, weight, bias, out, M, N, K, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1))
    return out
```

---

## Fused Add + LayerNorm

**Use when:** Post-attention residual add + normalization.
**Expected speedup:** 1.5-2x vs `F.layer_norm(x + residual, ...)`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, residual_ptr, out_ptr, weight_ptr, bias_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start = row_idx * n_cols

    # Load and add
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x = x + residual

    # LayerNorm
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    x_norm = x_centered / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    out = x_norm * weight + bias

    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


def fused_add_layernorm(
    x: torch.Tensor, residual: torch.Tensor,
    weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_contiguous() and residual.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    residual = residual.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    grid = (n_rows,)
    fused_add_layernorm_kernel[grid](x, residual, out, weight, bias, n_rows, n_cols, eps)
    return out.view(shape)
```

---

## Pattern Selection Guide

| Use Case | Pattern | Expected Speedup |
|----------|---------|------------------|
| FFN activation + dropout | GELU + Dropout | 1.8-2.2x |
| LLaMA FFN gate | SiLU + Multiply | 1.5-2x |
| LLaMA norm | RMSNorm | 1.4-2x |
| FFN with activation | Linear + GELU | 1.3-1.6x |
| Post-attention | Add + LayerNorm | 1.5-2x |
