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

# Triton Core API Reference

## triton.jit

Decorator that JIT-compiles a function into a GPU kernel using the Triton compiler.

### Signature

```python
@triton.jit  # simple form — no parens needed
@triton.jit(do_not_specialize=None, do_not_specialize_on_alignment=None,
            debug=None, noinline=None, repr=None, launch_metadata=None)
```

| Param | Type | Purpose |
|-------|------|---------|
| `do_not_specialize` | `Iterable[int\|str]\|None` | Args to skip value-specialization (by index or name) |
| `do_not_specialize_on_alignment` | `Iterable[int\|str]\|None` | Args to skip alignment-specialization |
| `debug` | `bool\|None` | Enable interpreter mode / debug prints |
| `noinline` | `bool\|None` | Prevent inlining when called from another jit'd function |

### Implicit Pointer Conversion

Objects with both `.data_ptr()` and `.dtype` (e.g., PyTorch tensors) are auto-converted
to device pointers. You never call `.data_ptr()` yourself in the launch call:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

# Launch — pass tensors directly, NOT x.data_ptr()
add_kernel[(grid,)](x, y, out, x.numel(), BLOCK=1024)
```

### Specialization Rules

Triton recompiles a kernel when argument properties change. For each argument:

| Arg type | Specialized on | Effect |
|----------|---------------|--------|
| Pointer (tensor) | 16-byte alignment of `.data_ptr()` | Enables vectorized loads/stores |
| Integer scalar | Whether value == 1 | Dead-code elimination for guards |
| Integer scalar | Whether value is divisible by 16 | Enables optimized indexing |
| `tl.constexpr` | Exact value | Baked into compiled code as literal |

**Gotcha:** Each unique specialization signature triggers a full recompile. If a size arg
oscillates between aligned/unaligned values, you get 2 cached versions (fine). But if you
pass truly random integers, use `do_not_specialize` to avoid cache explosion:

```python
@triton.jit(do_not_specialize=["stride_x"])
def my_kernel(x_ptr, stride_x, BLOCK: tl.constexpr):
    ...
```

### constexpr Parameters

Annotate with `tl.constexpr` to make a param a compile-time constant. Required for
values used in `tl.arange()`, `tl.zeros()`, tensor shapes, and `tl.static_assert`.
Each distinct value triggers recompilation.

```python
@triton.jit
def kernel(x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    ...
```

---

## triton.autotune

Decorator that benchmarks multiple `triton.Config`s and caches the fastest per key.

### Signature

```python
@triton.autotune(
    configs: list[triton.Config],
    key: list[str],
    prune_configs_by: dict | None = None,
    reset_to_zero: list[str] | None = None,
    restore_value: list[str] | None = None,
    warmup: int = 25,
    rep: int = 100,
    use_cuda_graph: bool = False,
)
```

| Param | Purpose |
|-------|---------|
| `configs` | List of `triton.Config` objects to benchmark |
| `key` | Arg names whose values form the cache key (e.g., `["M", "N", "K"]`) |
| `prune_configs_by` | Dict with `early_config_prune`, `perf_model`, `top_k` to reduce search space |
| `reset_to_zero` | Arg names zeroed before each config trial (for accumulator correctness) |
| `restore_value` | Arg names restored to original value after each trial |
| `warmup` | Warmup time in ms per config (default 25) |
| `rep` | Benchmark time in ms per config (default 100) |

### Example

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
    reset_to_zero=["c_ptr"],  # zero output buffer between trials
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

### Debugging Autotuning

```bash
TRITON_PRINT_AUTOTUNING=1 .venv/bin/python my_script.py
```

Prints the winning config and total tuning time per kernel to stdout.

### prune_configs_by Details

```python
def early_prune(configs, named_args, **kwargs):
    """Drop configs that exceed shared memory or have bad aspect ratios."""
    M, N = named_args["M"], named_args["N"]
    return [c for c in configs if c.kwargs["BLOCK_M"] <= M and c.kwargs["BLOCK_N"] <= N]

@triton.autotune(
    configs=[...],
    key=["M", "N"],
    prune_configs_by={"early_config_prune": early_prune},
)
```

Fields: `early_config_prune(configs, named_args, **kwargs) -> list[Config]`,
`perf_model(named_args, config, **kwargs) -> float` (estimated time),
`top_k` (int, keep only top_k configs after perf_model ranking).

### Gotchas

- `reset_to_zero` is critical for kernels that accumulate (e.g., matmul output).
  Without it, later configs see leftover values from earlier trials.
- Autotuning happens on first call with each unique key combination. Subsequent calls
  with the same key values use the cached winner.
- Decorator order: `@triton.autotune` must be the outermost, then `@triton.heuristics`
  (if used), then `@triton.jit` innermost.

---

## triton.Config

Represents one candidate configuration for `triton.autotune`.

### Signature

```python
triton.Config(
    kwargs: dict[str, Any],
    num_warps: int = 4,
    num_stages: int = 3,
    num_ctas: int = 1,
    maxnreg: int | None = None,
    pre_hook: Callable | None = None,
)
```

| Param | Default | Purpose |
|-------|---------|---------|
| `kwargs` | (required) | Dict mapping `tl.constexpr` param names to values |
| `num_warps` | 4 | Threads per block = `num_warps * 32` |
| `num_stages` | 3 | Software pipelining depth for global loads |
| `num_ctas` | 1 | Cooperative thread arrays (multi-CTA kernels, Hopper+) |
| `maxnreg` | None | Max registers per thread (trades occupancy vs spilling) |
| `pre_hook` | None | `fn(args: dict)` called before kernel launch |

### Example with pre_hook

```python
def zero_output(args):
    """Zero the output tensor before the kernel runs."""
    args["c_ptr"].zero_()

triton.Config(
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    num_warps=4,
    num_stages=3,
    pre_hook=zero_output,
)
```

### Tuning Guidance

| Parameter | Small tiles / low occupancy | Large tiles / high throughput |
|-----------|----------------------------|-------------------------------|
| `num_warps` | 2-4 | 8-16 |
| `num_stages` | 2 (less shared mem) | 3-5 (hide global latency) |
| `maxnreg` | None (let compiler decide) | 128-255 (force occupancy) |

**Gotcha:** `num_stages > 1` requires shared memory for buffering. Large tiles +
many stages can exceed shared memory limits, causing silent fallback or launch failure.

### GPU-Specific Config Guidelines

**H100 (Hopper):** HBM3, 168 SMs, large shared memory.

- Prefer larger blocks (1024-4096), more warps (8-16), `num_stages=4+`.

**A100 (Ampere):** Balanced config.

- Block sizes 512-2048, `num_stages=3` typically optimal.

**V100 (Volta):** Less shared memory.

- Smaller blocks (256-1024), fewer stages (2), warps 4-8.

---

## triton.heuristics

Decorator that computes meta-parameters from kernel arguments at launch time,
avoiding the cost of autotuning for values that can be derived deterministically.

### Signature

```python
@triton.heuristics(values: dict[str, Callable])
```

`values` maps constexpr parameter names to functions. Each function receives
the kernel's named arguments as a dict and returns the computed value.

### Example

```python
@triton.heuristics(
    values={
        "BLOCK_SIZE": lambda args: triton.next_power_of_2(args["n_cols"]),
        "num_warps": lambda args: 4 if args["n_cols"] <= 1024 else 8,
    }
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    ...

# Launch — BLOCK_SIZE is computed automatically, not passed
softmax_kernel[(n_rows,)](x, out, n_cols)
```

### Combined with autotune

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])}
)
@triton.jit
def kernel(x_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

**Required order (outermost to innermost):** `@autotune` -> `@heuristics` -> `@jit`

### triton.next_power_of_2

```python
triton.next_power_of_2(n)  # 7 -> 8, 8 -> 8, 1000 -> 1024
```

Host-side utility. Common pattern: derive BLOCK_SIZE from a problem dimension
so the block covers the full row/column in one pass.

### Gotchas

- Heuristic functions run on the host (CPU) at every kernel launch, not on GPU.
- `@heuristics` must come AFTER `@autotune` but BEFORE `@jit` in decorator stack.
- Values computed by heuristics override any same-named values in `triton.Config.kwargs`.
- Returning non-power-of-2 for a `BLOCK_*` param is valid but usually suboptimal.

---

## Decorator Stacking Summary

```
@triton.autotune(...)      # outermost — optional
@triton.heuristics(...)    # middle — optional
@triton.jit                # innermost — required
def kernel(...):
```

| Combo | Use case |
|-------|----------|
| `@jit` only | Fixed config, simplest kernels |
| `@autotune` + `@jit` | Search over tile sizes and hardware params |
| `@heuristics` + `@jit` | Derive config from args, no search needed |
| `@autotune` + `@heuristics` + `@jit` | Search some params, derive others |
