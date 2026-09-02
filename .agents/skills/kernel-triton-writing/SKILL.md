---
name: kernel-triton-writing
description: >
  ONLY for OpenAI Triton (@triton.jit) kernel development. NEVER use for
  CUDA C++ kernels, TileIR, or profiling tools such as ncu or nsys. Use when
  the request explicitly involves implementing, reviewing, or debugging a
  Triton kernel in vLLM.
license: Apache-2.0
metadata:
  source: https://github.com/NVIDIA/TensorRT-LLM
  source_commit: 395985c025c8d1cf5aa842bc752b337ba88721b6
---

# Triton Kernel Writing

<!--
The initial draft was copied from NVIDIA TensorRT-LLM's kernel-triton-writing
skill at commit 395985c025c8d1cf5aa842bc752b337ba88721b6. The content has
since been substantially rewritten for vLLM. See ORIGIN.md for provenance.
-->

Use this workflow for OpenAI Triton (`@triton.jit`) work in vLLM. Use the
`kernel-microbenchmark` skill as well when performance measurement or generated
code inspection is part of the task.

## 1. Confirm the fit

If Triton was explicitly requested, honor that choice. Otherwise, first inspect
nearby vLLM implementations and decide whether Triton is appropriate. Compare it
with existing vLLM operators, PyTorch compilation, and maintained vendor or
third-party kernels. Fusion potential alone does not guarantee a speedup, and a
standalone operation is not automatically a poor Triton candidate.

Record the intended devices, dtypes, layouts, shape distribution, numerical
contract, and whether compilation or autotuning latency matters. Unless support
follows an existing vLLM compatibility contract, do not claim backend or device
support without relevant test coverage.

## 2. Design around the contract

- Define which program owns each output or whether an atomic update is required.
  Make pointer arithmetic and strides explicit; do not assume inputs are
  contiguous unless the public contract does.
- Mask every potentially out-of-bounds load and store. Select masked-load
  values that are neutral for the operation, such as zero for a sum or negative
  infinity for a floating-point maximum.
- Remember that `tl.where` evaluates both branches. Use load/store masks when a
  branch must prevent a memory access.
- Choose accumulator and intermediate dtypes from the algorithm's numerical
  requirements. Promotion is operation-specific: for example, reductions and
  `tl.dot` have their own accumulation rules. Do not apply a blanket rule that
  every math function requires fp32.
- `tl.store` converts values to the pointer element type. Cast explicitly when
  it documents a deliberate rounding point, not because every store requires
  one.
- Keep index calculations non-negative when possible. Triton integer division
  and remainder can differ from Python for negative tensor operands; consult
  [semantics.md](references/semantics.md) when porting signed index math.
- Treat block sizes, warp counts, stage counts, and launch order as tuning
  choices, not GPU-family rules. Constraints on a specific operation, such as
  `tl.arange`, do not imply that every meta-parameter must be a power of two.
- Avoid device-to-host scalar extraction such as `.item()` in a hot wrapper
  when the source is a device tensor. It can synchronize the host and device.
  Do not replace a real random seed with a pointer-derived value; preserve the
  operator's RNG and determinism contract.

Use a tuple launch grid when it is static. Use a callable grid only when it must
depend on compile-time meta-parameters, including autotuned values.

## 3. Implement in vLLM

Match the nearest vLLM module's public interface, dispatch, platform guards,
device handling, and style. Prefer extending an existing implementation over
creating a parallel abstraction.

A basic one-dimensional kernel has this shape:

```python
@triton.jit
def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x  # Replace with the operation.
    tl.store(out_ptr + offsets, result, mask=mask)
```

This is a structural example, not a recommended block size or complete public
wrapper. For GEMM, attention, persistent kernels, tensor descriptors, or other
specialized designs, start from a current official Triton tutorial and adapt it
to the installed Triton version and vLLM conventions. Do not copy experimental
APIs without checking that vLLM's supported Triton versions expose them.

### Autotuning

Use `triton.autotune` only when its runtime cost and cache behavior fit the
deployment path. Fixed configurations, heuristics, or an existing vLLM tuning
mechanism may be preferable.

When autotuning a kernel that mutates a buffer, ensure every candidate sees the
same initial state. Use the installed Triton version's `reset_to_zero`,
`restore_value`, or hooks as appropriate. A normal matmul that overwrites its
output does not need `reset_to_zero` merely because it uses an accumulator
internally.

Do not encode generic H100/A100/V100 recipes. SKU resources, shapes, dtypes,
compiler versions, and register pressure all affect the best configuration.
Measure representative production shapes on each supported target.

## 4. Verify correctness

Extend the nearest existing pytest suite, normally under `tests/kernels/`.
Before writing tests, identify the public behavior, failure mode, and smallest
test level that catches it.

Cover the dimensions relevant to the contract:

- empty or minimum supported sizes and non-divisible tile boundaries;
- representative production shapes, including awkward dimensions;
- supported dtypes and layouts, including non-contiguous inputs if promised;
- aliasing, in-place behavior, RNG state, and determinism when applicable;
- numerical edge cases such as large magnitudes, zeros, infinities, or NaNs
  when the operator defines behavior for them.

Compare the public wrapper with an independent reference. Derive tolerances
from the dtype, operation, reduction depth, and documented precision mode; do
not use a universal tolerance table. For matmul-like operations, configure the
reference and Triton kernel to use comparable input and accumulation precision.

Run the focused suite through the repository environment:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

Do not use benchmark agreement as a substitute for a correctness test.

## 5. Measure only after correctness passes

When performance is in scope, follow `$kernel-microbenchmark`. Put durable
kernel benchmarks under `benchmarks/kernels/` and report distributions,
representative shapes, hardware, software versions, and benchmark conditions.
Compare end-to-end cost when compilation, autotuning, allocations, or wrapper
overhead can affect the user-visible result.

Treat a slowdown or regression as evidence to investigate, not proof that the
reference is optimal or Triton is unsuitable. Inspect generated code and
resource use when the benchmark warrants it.

## 6. Debug systematically

Reduce failures to a small shape, separate compilation failures from numerical
errors and memory faults, and use the tools in
[troubleshooting.md](references/troubleshooting.md). Never delete a broad or
unresolved cache path. If cache invalidation is justified, first resolve and
confirm the exact Triton cache directory, then move that directory aside so it
can be restored.

## Current authoritative references

Check these before relying on signatures, backend support, or experimental
features:

- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)
- [Official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [vLLM kernel benchmarks](../../../benchmarks/kernels/)

Consult [semantics.md](references/semantics.md) for the few semantic hazards
worth keeping local. Consult [troubleshooting.md](references/troubleshooting.md)
for a compact debugging checklist. Prefer current official documentation and
the installed API over copied signature catalogs.
