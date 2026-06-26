# Mojo W4A16 GEMM for ROCm

This directory contains the source-controlled Mojo W4A16 GEMM used by
`vllm.model_executor.kernels.linear.mixed_precision.mojo_w4a16`.

`runtime.py` owns the portable Python-side runtime for these kernels: policy
lookup, variant selection, generated extension materialization, runner caching,
PyTorch-owned split-K scratch preparation, and HIP-stream launches. The vLLM
linear integration imports that runtime by path, then keeps only torch custom
op registration and AWQ layer plumbing.

The runtime generates one direct Mojo Python extension per selected tuned policy
bucket. Each extension exposes `W4A16Runner`, which owns:

- a Mojo `DeviceContext`;
- a compiled W4A16 variant kernel handle;
- a compiled split-K reducer handle and partial buffer when the selected
  variant needs them.

The exported runner API is intentionally variant-opaque to vLLM. Python passes
the same tensor set for every variant: output, activations, original AWQ-packed
weights, K-packed weights, qzeros, and scales. Ring variants ignore the
K-packed tensor; split-K variants ignore the original packed tensor.

During CUDA/HIP graph capture, the torch op only wraps tensor data pointers and
calls the HIP launch shim. The shim passes the current PyTorch HIP stream to
the exported Mojo launch symbol, and Mojo enqueues the already compiled handle
on that active stream. It does not compile Mojo kernels, load modules, or
synchronize during capture.

## Opt-in Usage

This kernel is not selected by `linear_backend="auto"`. To use it in vLLM,
install the local runtime requirements, make sure `max` and `mojo` are
available on `PATH`, then select the Mojo linear backend:

```bash
pip install -r vllm/csrc/rocm/mojo_gemm_w4a16/requirements.txt
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
  --trust-remote-code \
  --linear-backend mojo
```

Dependency checks are lazy and only run when the Mojo W4A16 kernel is selected
with `--linear-backend mojo`.

The opt-in backend currently targets ROCm RDNA3/RDNA3.5 devices (`gfx1100` and
`gfx1151`) with fp16/bf16 activations and 4-bit WNA16 weights supported by the
vLLM mixed-precision linear path.

## Tuning And Test

Runtime policy files are JSON files keyed like other vLLM tuned configs and use
explicit fields such as `tile_M`, `warps_N`, and `splitk_block_K`.

Tune a model and write serving-ready JSON policy files directly with:

```bash
python vllm/csrc/rocm/mojo_gemm_w4a16/benchmark_w4a16.py --tune \
  --model QuantTrio/Qwen3.5-9B-AWQ \
  --trust-remote-code \
  --save-dir vllm/csrc/rocm/mojo_gemm_w4a16/policies \
  --target-accelerator gfx1151
```

Run the single direct-op test for policy loading and Mojo execution:

```bash
python vllm/csrc/rocm/mojo_gemm_w4a16/test_policy_direct_op.py \
  --m 1 --n 4096 --k 12288 --group-size 128 \
  --expected-variant kpacked_dot2
```

## Source map

- `mojo/common.mojo`: compile-time config, layouts, benchmark test data helpers.
  Policy `variant` is the single source of truth for generated kernel selection.
- `mojo/kernel_common.mojo`: shared dequantization, LDS staging, WMMA/fdot helpers,
  and accumulator stores.
- `mojo/ring_buffer.mojo`: ring-buffer synchronization helpers.
- `mojo/splitk_reduce.mojo`: shared reducers for split-K variants.
- `mojo/kernels/ring_ab_staged.mojo`: full A+B staged ring-buffer variant.
- `mojo/kernels/ring_b_staged.mojo`: B-only staged ring-buffer variant.
- `mojo/kernels/b_staged_sync.mojo`: single-stage synchronized B-only variant.
- `mojo/kernels/kpacked_dot2_splitk.mojo`: K-packed split-K fdot2 variant.
- `mojo/kernels/kpacked_wmma16_splitk.mojo`: K-packed split-K WMMA16 variant.
- `runtime.py`: Python policy selection, generated extension build/load,
  runner cache, graph-safe scratch preparation, and launch helpers.
- `benchmark_w4a16.py`: standalone tuner/benchmark entrypoint that emits
  serving-ready JSON policy files directly into `policies/`.
- `benchmark_runner.mojo`: single-config Mojo benchmark runner used by the
  tuner. It imports the same packaged variant sources used by `runtime.py`.
- `templates/direct_extension_template.mojo`: generated Python/native runner wrapper.
- `policies/`: bundled tuned JSON policy files keyed by N, K,
  group size, vLLM device name, dtype, and op. Each file maps M buckets to
  explicit tile/warp/split-K fields such as `tile_M`, `warps_N`, and
  `splitk_block_K`.
- `test_policy_direct_op.py`: direct public-op test for policy loading and Mojo
  custom op execution.
- `hip_launch_shim.cpp`: PyTorch C++ shim that passes the active HIP stream.
