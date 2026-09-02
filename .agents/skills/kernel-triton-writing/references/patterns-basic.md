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

# Triton Basic Kernel Patterns

Reusable patterns extracted from the official Triton tutorials.
Each section contains a complete kernel, its launch wrapper, and annotations.

---

## Vector Addition

The simplest Triton pattern: 1D parallel map over contiguous data.

### Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # compile-time constant: controls tile width
):
    # Each program instance owns one tile of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    # Compute the start offset for this program's tile, then the per-lane offsets.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard against out-of-bounds access on the final, possibly partial tile.
    mask = offsets < n_elements
    # Load inputs from DRAM — masked lanes get a safe default (0.0).
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write result back — only masked lanes write.
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Launch Wrapper

```python
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # Grid is a callable: Triton passes meta-parameters (incl. BLOCK_SIZE) at launch.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Benchmark Pattern

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

### Key Takeaways

- **Offset pattern:** `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` — the universal 1D tiling idiom.
- **Masking:** Always guard the last tile with `offsets < n_elements`.
- **Grid as callable:** `lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)` lets autotune vary BLOCK_SIZE.
- **Pointer arithmetic:** Triton pointers support `ptr + offset_tensor` for vectorized addressing.

---

## Fused Softmax

Row-wise softmax fused into a single kernel. Fusion reduces DRAM traffic from
`5*M*N + 2*M` bytes (naive: read 3x for max/exp/sum, write 2x) down to `M*N`
read + `M*N` write by keeping intermediate results in SRAM.

### Kernel

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,   # must be >= n_cols (padded to power-of-2)
    num_stages: tl.constexpr,   # software pipelining depth
):
    # Persistent-kernel style: each program processes multiple rows, strided.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Compute pointers for this row.
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Mask: BLOCK_SIZE is rounded up to power-of-2, so some lanes are OOB.
        mask = col_offsets < n_cols
        # Load row; OOB lanes get -inf so they don't affect max/sum.
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # --- Numerical stability: subtract row-max before exp ---
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Store result.
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

### Launch Wrapper

```python
def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    # BLOCK_SIZE must cover the full row — round up to power-of-2.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Heuristic: use more warps for wider rows.
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8
    # Persistent kernel: launch fewer programs than rows for large inputs.
    # Each SM can run ~4 programs concurrently (occupancy dependent).
    num_stages = 4 if BLOCK_SIZE > 2048 else 2
    y = torch.empty_like(x)
    # Grid: one dimension, capped by number of rows.
    num_programs = min(n_rows, 1024)  # cap to avoid over-subscription
    softmax_kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y
```

### Key Takeaways

- **Numerical stability:** Always `row - tl.max(row, axis=0)` before `tl.exp`.
- **Power-of-2 padding:** `BLOCK_SIZE = triton.next_power_of_2(n_cols)` with `-inf` masking for OOB lanes.
- **Persistent kernel:** `tl.range(start, end, step, num_stages=...)` loops over multiple rows per
  program, improving occupancy and enabling software pipelining.
- **Fusion benefit:** One kernel replaces three separate passes (max, exp/sum, divide), keeping
  all intermediates in registers/SRAM instead of round-tripping through DRAM.

---

## Low-Memory Dropout

Traditional dropout stores a full-size bit mask. This pattern stores only an `int32` seed and
recomputes the mask on-the-fly via Triton's built-in PRNG. The same seed + offsets produce
identical random values, so forward and backward passes see the same mask without storing it.

### Kernel

```python
@triton.jit
def _seeded_dropout(
    x_ptr, output_ptr,
    n_elements,
    p,              # dropout probability (float, 0 to 1)
    seed,           # int32 seed — the ONLY state needed to reproduce the mask
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tl.rand: deterministic PRNG — given the same (seed, offsets), produces
    # the same uniform float32 values in [0, 1). No global state needed.
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # Scale kept elements by 1/(1-p) so expected value is unchanged (inverted dropout).
    # Dropped elements become 0.0.
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Launch Wrapper

```python
def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output
```

### Key Takeaways

- **Memory savings:** State = 1 `int32` seed, not an `(N,)` bool tensor.
- **Deterministic PRNG:** `tl.rand(seed, offsets)` is pure-functional — same inputs, same outputs.
- **Inverted dropout:** `x / (1 - p)` scales at train time so inference needs no adjustment.
- **`tl.where` pattern:** `tl.where(cond, val_true, val_false)` is the standard Triton conditional —
  works element-wise on block tensors, compiles to predicated instructions (no branch divergence).
