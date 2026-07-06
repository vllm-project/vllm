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

## Benchmarking methodology (IMPORTANT)

All numbers below are **sole-process, sequential, same GPU (gpu0), nothing else
running**. On this shared 2×H100 node, running comparisons in parallel (gpu0 +
gpu1) or alongside other GPU work (downloads, builds, profilers) produces large
**contention artifacts** — an earlier draft reported a spurious "2.8× Helion
slowdown" that was entirely GPU contention and vanished under clean measurement.
Never benchmark in parallel here. input 256 / output 128 / batch 16, warmup 5 /
iters 20.

## Results (2×H100, Qwen3-1.7B-FP8, block-fp8)

| configuration | P50 | Avg | note |
|---|---|---|---|
| **DeepGEMM ON (default)** | **0.363 s** | 0.363 s | Helion not used on this path |
| DeepGEMM OFF + Helion | 0.448 s | 0.515 s | P99 1.54 s = one-time JIT spike |
| DeepGEMM OFF + native | 0.466 s | 0.463 s | |

Reading it:
- **The default DeepGEMM path (0.363 s) is fastest** and does not use Helion at
  all: for block-fp8, DeepGEMM fuses the activation quant *into* the GEMM, so the
  standalone `_C` quant ops never exist and `HelionKernelSwapPass` swaps 0.
- On the (slower) `VLLM_USE_DEEP_GEMM=0` path where Helion does engage
  (`_helion_rms_norm_per_block_quant` + `_helion_per_token_group_fp8_quant`),
  **Helion is ~4% faster than native at P50 (0.448 vs 0.466)** — but its **Avg is
  worse (0.515)** because the Helion Triton kernels JIT lazily and cause a one-time
  P99 spike (1.54 s). A warmup pass that pre-JITs the kernels would remove that.
- Net: on block-fp8 the default DeepGEMM path already wins; forcing the
  DeepGEMM-off path just to use Helion is not worthwhile.

## RedHatAI/Qwen3-1.7B-FP8-dynamic (compressed-tensors W8A8)

Different scheme: `compressed-tensors`, per-channel static fp8 weights + **per-token
dynamic** fp8 activations, **no** `weight_block_size`. So:

- **Does not use DeepGEMM.** The GEMM is CUTLASS scaled_mm
  (`CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8`). The
  "DeepGEMM enabled" startup lines are capability probes, not usage.
- **Does not use Helion by default either.** For this config vLLM leaves the
  fusion passes off (`custom_ops=['none']`, `fuse_norm_quant=False`,
  `fuse_act_quant=False`), so the fused `_C` quant ops are never created — the
  activation quant folds into the cutlass linear. `HelionKernelSwapPass` runs but
  swaps 0. ON == OFF: both P50 ≈ 0.348 s.
- **Helion *can* engage if fusion is forced on:**
  `--compilation-config '{"mode":3,"custom_ops":["+quant_fp8"],"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}'`
  — then it swaps `rms_norm_dynamic_per_token_quant` +
  `dynamic_per_token_scaled_fp8_quant` (the per-token ops Helion was designed for)
  and `_helion_*` kernels execute. Clean numbers (sole-process):

  | configuration | P50 | Avg |
  |---|---|---|
  | **default (fusion off, CUTLASS)** | **0.349 s** | 0.349 s |
  | forced-fusion + Helion | 0.360 s | 0.360 s |
  | forced-fusion + native | 0.365 s | 0.365 s |

  So forced-fusion + Helion (0.360 s) is **marginally faster than forced-fusion +
  native (0.365 s)** and only slightly slower than the default fusion-off path
  (0.349 s, the fusion overhead). It is **not** a large loss — the earlier
  "0.976 s / 2.8× slower" was a GPU-contention artifact (see methodology). No JIT
  spike here (P99 0.369 s).

## Qwen3-8B-FP8 and Qwen3-32B-FP8

Same quantization scheme as 1.7B — `fp8 e4m3, weight_block_size=[128,128],
activation_scheme=dynamic` (block-fp8, group 128). Therefore they take the **same
DeepGEMM path by default** and, like 1.7B, do **not** use the Helion kernels unless
`VLLM_USE_DEEP_GEMM=0`. Helion configs are tuned for their hidden sizes
(8B: 4096, 32B: 5120 — both in the tuned `hidden_size_list`), so the swap would
engage on the DeepGEMM-off path, with the same DeepGEMM-wins caveat.

## Kernel-level microbenchmark (isolated, CUDA-graph capture, hidden=2048)

The kernels themselves are substantially faster; the end-to-end effect is diluted
by the rest of the forward pass.

| op | decode (1–256 tok) | prefill (4096 tok) |
|---|---|---|
| rms_norm_dynamic_per_token_quant | 1.35–1.44× | 4.09× |
| dynamic_per_token_scaled_fp8_quant | 1.21–1.25× | 1.42× |

(Measured under CUDA-graph replay — eager mode shows a flat ~30 µs custom-op Python
dispatch floor that cudagraphs/compile remove, so do not benchmark these in eager.)

## Overall conclusion

- The Helion integration is **correct**: ops are cudagraph-captured, kernels run,
  and in isolation they are 1.2–4× faster than native.
- End-to-end on these Qwen3 FP8 variants, Helion is **roughly neutral** — never a
  clear win, never a real regression:
  - block-fp8: default DeepGEMM (0.363 s) wins and doesn't use Helion; on the
    DeepGEMM-off path Helion is ~4% faster than native at P50 but its Avg is hurt
    by JIT spikes.
  - W8A8-dynamic: default fusion-off CUTLASS (0.349 s) is fastest; forcing fusion
    to reach Helion (0.360 s) ≈ forced-fusion native (0.365 s).
- The earlier "8.85× nvjet GEMM regression / 2.8× slowdown / GPU idle / broken
  run-ahead" narrative was a **GPU-contention measurement artifact** from running
  benchmarks in parallel; it does not reproduce under sole-process measurement.
  ncu confirmed all kernels (including the bf16 lm_head nvjet GEMM) are healthy.
- Lowest-hanging improvement: a **warmup pass** to pre-JIT the Helion Triton
  kernels before cudagraph capture, removing the one-time P99 spikes seen on the
  block-fp8 DeepGEMM-off path.

**Overall:** Helion is not a net win for any Qwen3-1.7B FP8 variant tested.
