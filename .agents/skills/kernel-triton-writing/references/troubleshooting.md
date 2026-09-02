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

# Triton Troubleshooting, Debugging, and Benchmarking

## Debug Operations — Compile-Time

### static_print — Inspect Types and Constants at Compile Time

Prints values during kernel compilation (not at runtime). Use to verify
constexpr values, tensor shapes, and dtypes.

```python
@triton.jit
def kernel(X_ptr, N, BLOCK_SIZE: tl.constexpr):
    tl.static_print("BLOCK_SIZE", BLOCK_SIZE)           # prints the constexpr value
    x = tl.load(X_ptr + tl.arange(0, BLOCK_SIZE))
    tl.static_print("x dtype", x.dtype)                 # prints the tensor dtype
    tl.static_print("x shape", x.shape)                 # prints the tensor shape
```

Output appears in stderr during compilation (not on device):

```
BLOCK_SIZE 1024
x dtype float32
x shape (1024,)
```

### static_assert — Compile-Time Invariant Checks

Fails compilation if condition is false. Use for constexpr guards.

```python
@triton.jit
def kernel(X_ptr, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of 32")
    tl.static_assert(BLOCK_SIZE <= 4096, "BLOCK_SIZE too large")
```

| Function | Runs When | Requires TRITON_DEBUG | Use For |
|----------|-----------|----------------------|---------|
| `tl.static_print(label, value)` | Compilation | No | Inspecting types, shapes, constexpr values |
| `tl.static_assert(cond, msg)` | Compilation | No | Enforcing constexpr constraints |

---

## Debug Operations — Runtime (On-Device)

### device_print — Print Tensor Values from GPU

Prints values at runtime from every active thread. Produces large output
on multi-element blocks; use masks or conditions to limit output.

```python
@triton.jit
def kernel(X_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs)

    # Print from all programs — very verbose
    tl.device_print("x", x)

    # Print from only program 0 — much less output
    if pid == 0:
        tl.device_print("x[0]", x)
```

### device_assert — Runtime Assertions (Requires TRITON_DEBUG=1)

Only executes when `TRITON_DEBUG=1` is set. Silent otherwise.

```python
@triton.jit
def kernel(X_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.device_assert(offs < N, "out-of-bounds access")
    x = tl.load(X_ptr + offs)
```

```bash
# Enable device_assert checks
TRITON_DEBUG=1 .venv/bin/python my_kernel.py
```

| Function | Runs When | Requires TRITON_DEBUG | Use For |
|----------|-----------|----------------------|---------|
| `tl.device_print(label, value)` | Runtime (GPU) | No | Inspecting tensor values on device |
| `tl.device_assert(cond, msg)` | Runtime (GPU) | **Yes** (`=1`) | Bounds checks, NaN guards, invariants |

### Gotcha: device_assert Does Nothing Without TRITON_DEBUG

If you add `tl.device_assert` and your kernel still silently produces wrong results,
check that `TRITON_DEBUG=1` is exported **before** the kernel is compiled/cached.

---

## Interpreter Mode — CPU Step-Through Debugging

Setting `TRITON_INTERPRET=1` runs all Triton kernels on the CPU using NumPy,
bypassing GPU compilation entirely. This enables standard Python debugging.

### Basic Usage

```bash
TRITON_INTERPRET=1 .venv/bin/python my_kernel.py
```

### Debugging with pdb

```bash
TRITON_INTERPRET=1 .venv/bin/python -m pdb my_kernel.py
```

Set breakpoints inside `@triton.jit` functions — they execute as normal Python
in interpreter mode.

```python
@triton.jit
def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs)
    # In interpreter mode, you can set a pdb breakpoint here
    # and inspect x as a numpy array
    import pdb; pdb.set_trace()   # works only with TRITON_INTERPRET=1
    y = x * 2
    tl.store(Y_ptr + offs, y)
```

### Interpreter Mode Limitations

| Limitation | Detail |
|------------|--------|
| No bfloat16 support | NumPy lacks native bf16; operations may error or use fp32 fallback |
| No indirect memory access | Gather/scatter patterns may not work correctly |
| No GPU-specific behavior | Race conditions, warp-level ops not simulated |
| Performance | Orders of magnitude slower than GPU — use small inputs only |
| Caching | Set `TRITON_INTERPRET=1` before any kernel is compiled/cached |
| atomic_add with fp16 | Known issue — may raise `ValueError('unsupported data type')` |

---

## Third-Party Debug Tools

| Tool | Vendor | Purpose | Usage |
|------|--------|---------|-------|
| `compute-sanitizer` | NVIDIA | Memory access checker (out-of-bounds, races) | `compute-sanitizer .venv/bin/python my_kernel.py` |
| `compute-sanitizer --tool memcheck` | NVIDIA | Detailed memory error reports | `compute-sanitizer --tool memcheck .venv/bin/python my_kernel.py` |
| `compute-sanitizer --tool racecheck` | NVIDIA | Shared memory race detection | `compute-sanitizer --tool racecheck .venv/bin/python my_kernel.py` |
| AddressSanitizer | AMD (ROCm) | Memory error detection on AMD GPUs | Compile with ASan flags |
| `triton-viz` | Community | Visual trace of memory access patterns | `uv pip install triton-viz` |

### compute-sanitizer Example

```bash
# Check for out-of-bounds memory access
compute-sanitizer --tool memcheck .venv/bin/python my_kernel.py

# Check for shared memory race conditions
compute-sanitizer --tool racecheck .venv/bin/python my_kernel.py
```

---

## Benchmarking — triton.testing

### do_bench — Micro-Benchmark a Function

```python
import triton

ms = triton.testing.do_bench(lambda: my_kernel[grid](x, y, N, BLOCK_SIZE=1024))
print(f"{ms:.3f} ms")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | required | Zero-arg function to benchmark (use lambda) |
| `warmup` | `int` | `25` | Warmup time in milliseconds |
| `rep` | `int` | `100` | Repetition time in milliseconds |
| `grad_to_none` | `torch.Tensor` | `None` | Reset this tensor's gradient to None each iteration |
| `quantiles` | `list[float]` | `None` | Percentiles to return (e.g., `[0.2, 0.5, 0.8]`) |
| `return_mode` | `str` | `"mean"` | `"min"`, `"max"`, `"mean"`, `"median"`, or `"all"` |

### Benchmark Class + perf_report — Parameterized Benchmarks

```python
import triton
from triton.testing import Benchmark, perf_report

@perf_report(
    Benchmark(
        x_names=["N"],                              # argument to vary
        x_vals=[2**i for i in range(10, 25)],       # values for N
        line_arg="provider",                         # line grouping
        line_vals=["triton", "torch"],               # line values
        line_names=["Triton", "PyTorch"],            # legend labels
        plot_name="vector-add-performance",          # plot filename
        args={},                                     # fixed args
        xlabel="Vector Size (N)",
        ylabel="GB/s",
        x_log=True,
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.randn(N, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: my_kernel[grid](x, y, output, N, BLOCK_SIZE=1024))
    else:
        ms = triton.testing.do_bench(lambda: x + y)
    gbps = 3 * x.numel() * x.element_size() / ms * 1e-6  # 3 = 2 reads + 1 write
    return gbps

# Run and save plot
benchmark.run(show_plots=True, save_path="./benchmarks/")
```

### Correctness Testing — torch.testing.assert_close

Triton does not ship its own `assert_close`. Use PyTorch:

```python
import torch
torch.testing.assert_close(triton_output, torch_reference, atol=1e-2, rtol=1e-2)
```

For fp16/bf16 kernels, use relaxed tolerances (`atol=1e-1, rtol=1e-1`).

---

## Common Errors Table

| Error / Symptom | Cause | Fix |
|-----------------|-------|-----|
| `shape mismatch` in binary op | Tensor shapes do not broadcast | Check shapes with `tl.static_print`; add `[:, None]` or `[None, :]` |
| `BLOCK_SIZE is not a constexpr` | Block size passed as runtime value | Add `: tl.constexpr` annotation to the parameter |
| `mask dimensions do not match` | Mask shape incompatible with load/store block | Ensure mask broadcasts to the pointer offset shape |
| OOM during autotuning | Too many `@triton.autotune` configs | Reduce config list; avoid combinatorial explosion of BLOCK_M/N/K |
| `device_assert` has no effect | `TRITON_DEBUG` not set to `1` | Export `TRITON_DEBUG=1` before running |
| Silent wrong results | Off-by-one in pointer arithmetic | Use `tl.device_print` to inspect offsets; test with `TRITON_INTERPRET=1` |
| `incompatible types` in store | Computed dtype does not match output pointer dtype | Cast explicitly: `tl.store(ptr, val.to(tl.float16))` |
| Kernel not updating after edit | Triton cache serving stale binary | Move the confirmed Triton cache directory aside and retry |
| `ValueError: unsupported data type` in interpreter | bf16 or fp8 used with `TRITON_INTERPRET=1` | Use fp16 or fp32 for interpreter debugging |
| `grid must be a tuple` | Lambda grid returns int, not tuple | Return `(value,)` with trailing comma |
| NaN output, correct logic | fp16 overflow in accumulator | Use `tl.float32` for accumulation, cast on store |
| `expected constexpr` in `tl.arange` | Non-constexpr argument to arange | Both args of `tl.arange(start, end)` must be constexpr |
| Mismatched results vs PyTorch | C integer division semantics | See concepts-semantics.md: Triton uses truncation, not floor division |
| `triton.OutOfResources` | Register/shared memory pressure | Reduce BLOCK_SIZE or number of live variables |

---

## Environment Variables Reference

| Variable | Value | Effect |
|----------|-------|--------|
| `TRITON_DEBUG` | `1` | Enable `device_assert`, extra runtime checks |
| `TRITON_INTERPRET` | `1` | Run kernels on CPU via NumPy (no GPU) |
| `TRITON_CACHE_DIR` | path | Override default cache directory (`~/.triton/cache/`) |
| `MLIR_ENABLE_DUMP` | `1` | Dump MLIR intermediate representations |
| `TRITON_PRINT_AUTOTUNING` | `1` | Print autotuning results to stderr |
