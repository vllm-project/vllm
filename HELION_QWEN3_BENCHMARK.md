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

## Results (2×H100, Qwen3-1.7B-FP8)

input 256 / output 128 / batch 16, warmup 10 / iters 30.

### Helion vs native, both on the `VLLM_USE_DEEP_GEMM=0` path

| metric | Helion ON | native OFF | delta |
|---|---|---|---|
| P50 latency | **0.444 s** | 0.465 s | **−4.5%** |
| P90 latency | **0.451 s** | 0.469 s | **−3.8%** |

Confirmed active in the ON run: `HelionKernelSwapPass` swaps the ops and the
`_helion_rms_norm_per_block_quant` / `_helion_per_token_group_fp8_quant` Triton
kernels execute; the pass is absent in the OFF run.

### vs the default DeepGEMM path — DeepGEMM wins

| configuration | P50 latency |
|---|---|
| **DeepGEMM ON (default)** | **0.368 s** |
| Helion, DeepGEMM off | 0.444 s |
| native, DeepGEMM off | 0.465 s |

**The default DeepGEMM path is ~17% faster than Helion-with-DeepGEMM-off.** For
Qwen3 block-fp8, DeepGEMM fuses the activation quant *into* the GEMM (eliminating
the standalone quant kernel entirely) and uses a highly tuned fp8 GEMM. Helion only
accelerates the separate norm/quant kernels, so disabling DeepGEMM to reach the
Helion path is a **net loss**. Helion would only be a win where quant is *not*
fused into the GEMM (e.g. per-token dynamic fp8 with a GEMM that consumes
pre-quantized activations, or fused rms_norm+dynamic-per-token-quant paths).

Notes:
- The Helion-vs-native ~4% gain is modest because norm+quant is a small share of
  total latency (GEMM + attention dominate).
- Helion Triton kernels JIT lazily per shape → a one-time P99 latency spike unless
  warmup covers every shape; amortized under sustained serving.

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
  and `_helion_*` kernels execute. But this is a large net loss vs the default —
  see "Forced-fusion Helion vs default" below.

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

### Forced-fusion Helion vs default (RedHatAI FP8-dynamic) — Helion loses badly

P50, matched config (input256/out128/bs16, warmup10/iters30):

| configuration | P50 latency |
|---|---|
| **default (fusion off, CUTLASS)** | **0.348 s** |
| fusion on + native | 0.363 s |
| fusion on + Helion | 0.976 s (~2.8x slower) |

Forcing fusion to reach Helion is a large **net loss**. The Helion kernels execute
but the run is consistently ~2.8x slower (P50 > Avg, i.e. steady, not JIT spikes) —
the Helion custom ops appear not to be captured into the full CUDA graph on this
path, so they pay the ~30 us/call custom-op dispatch overhead per layer per step.
(In isolated CUDA-graph capture the same kernels were 1.2-1.4x *faster*, so this is
a graph-capture/integration cost, not the kernel.)

**Overall:** Helion is not a net win for any Qwen3-1.7B FP8 variant tested.
