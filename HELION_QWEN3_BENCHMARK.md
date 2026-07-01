# Helion kernels for Qwen3-1.7B-FP8 — default-on + benchmark

## What this adds

The Helion kernels in `vllm/kernels/helion/ops` are registered as
`torch.ops.vllm_helion.*` custom ops but were previously never called. This change
wires them in as drop-in replacements for the native fused-quant CUDA ops, **on by
default**, toggleable with an env flag.

- `VLLM_USE_HELION_KERNELS` (default `1`) — set `0` to fall back to native kernels.
- `HelionKernelSwapPass` (`vllm/compilation/passes/fusion/helion_swap.py`) — a
  post-grad pass that retargets `auto_functionalized(torch.ops._C.<op>)` nodes to the
  `vllm_helion` equivalent in place. Safe because the swapped ops share the native
  op's argument names and mutated-arg layout, so node kwargs and `getitem` users
  stay valid. Runs last in `PostGradPassManager.configure()`, gated by
  `VLLM_USE_HELION_KERNELS and has_helion()`.

Swappable ops (identical schema to native): `rms_norm_dynamic_per_token_quant`,
`rms_norm_per_block_quant`, `dynamic_per_token_scaled_fp8_quant`,
`per_token_group_fp8_quant`. (`silu_mul_fp8` is skipped — different functional
signature.)

## Important: DeepGEMM caveat for Qwen3-1.7B-FP8

`Qwen/Qwen3-1.7B-FP8` uses **block-fp8 (group 128)**. On the default path
(`VLLM_USE_DEEP_GEMM=1`) the activation quantization is fused *inside* the DeepGEMM
block-scale GEMM op, so none of the Helion ops appear in the compiled graph and the
swap is a no-op. To exercise (and benchmark) the Helion kernels, run with
`VLLM_USE_DEEP_GEMM=0`, which surfaces `rms_norm_per_block_quant` and
`per_token_group_fp8_quant` as separate ops that the pass swaps to Helion.

## Environment prerequisites (2×H100 devvm)

- `nvcc` on PATH for DeepGEMM runtime JIT: `CUDA_HOME=/usr/local/cuda-13.0`,
  `PATH=/usr/local/cuda-13.0/bin:$PATH`.
- A `torchvision` stub, because `vllm/model_executor/warmup/kernel_warmup.py`
  unconditionally imports the MiniMax-M3 warmup (which imports torchvision). A
  minimal meta-path stub on `PYTHONPATH` is enough (never executed for Qwen3).

## Benchmark commands

```bash
MODEL=Qwen/Qwen3-1.7B-FP8   # or a local snapshot path

COMMON="--model $MODEL --input-len 256 --output-len 128 --batch-size 16 \
  --num-iters-warmup 10 --num-iters 30 -O3"

ENV="CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
  PYTHONPATH=/tmp/tvstub CUDA_VISIBLE_DEVICES=0 \
  VLLM_USE_DEEP_GEMM=0 VLLM_DISABLE_COMPILE_CACHE=1"

# Helion ON (default)
env $ENV VLLM_USE_HELION_KERNELS=1 vllm bench latency $COMMON

# Helion OFF (native baseline)
env $ENV VLLM_USE_HELION_KERNELS=0 vllm bench latency $COMMON
```

## Results (2×H100, Qwen3-1.7B-FP8, `VLLM_USE_DEEP_GEMM=0`)

input 256 / output 128 / batch 16, warmup 10 / iters 30:

| metric | Helion ON | native OFF | delta |
|---|---|---|---|
| P50 latency | **0.444 s** | 0.465 s | **−4.5%** |
| P90 latency | **0.451 s** | 0.469 s | **−3.8%** |

Confirmed active in the ON run: `HelionKernelSwapPass` swaps the ops and the
`_helion_rms_norm_per_block_quant` / `_helion_per_token_group_fp8_quant` Triton
kernels execute; the pass is absent in the OFF run.

Notes:
- The ~4% end-to-end gain is modest because norm+quant is a small share of total
  latency (GEMM + attention dominate).
- Helion Triton kernels JIT lazily per shape → a one-time P99 latency spike unless
  warmup covers every shape; amortized under sustained serving.

## Kernel-level microbenchmark (isolated, CUDA-graph capture, hidden=2048)

The kernels themselves are substantially faster; the end-to-end effect is diluted
by the rest of the forward pass.

| op | decode (1–256 tok) | prefill (4096 tok) |
|---|---|---|
| rms_norm_dynamic_per_token_quant | 1.35–1.44× | 4.09× |
| dynamic_per_token_scaled_fp8_quant | 1.21–1.25× | 1.42× |

(Measured under CUDA-graph replay — eager mode shows a flat ~30 µs custom-op Python
dispatch floor that cudagraphs/compile remove, so do not benchmark these in eager.)
