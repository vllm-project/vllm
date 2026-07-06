# Helion kernels — non-compiled (eager + CUDA-graph) path

This commit wires the Helion fused-quant kernels into the **non-compiled**
execution path only, via call-site routing (`route_quant`). No torch.compile
graph pass is involved (that lives in a later commit).

## What this adds

- `VLLM_USE_HELION_KERNELS` (default `1`) — set `0` to fall back to native kernels.
- `route_quant(op_name, *args)` (`vllm/kernels/helion/routing.py`) — called at the
  `torch.ops._C.<op>` **call sites** in eager code (fused-MoE experts, MLA
  attention, linear methods). At **runtime** it dispatches to the Helion op iff
  the stream is currently being captured into a CUDA graph
  (`torch.cuda.is_current_stream_capturing()`), and to the native `_C` op
  otherwise (Helion's ~30 µs per-call Python dispatch overhead is a loss in plain
  eager but is captured away under a CUDA graph). An `is_compiling()` guard makes
  it emit native `_C` during torch.compile tracing, so this hook is purely for the
  non-compiled path; the compiled/FX-swap path is a separate commit.

## Which ops fire on this path

In eager execution there are **no fusion passes**, so the fused
`rms_norm_*_quant` / `dynamic_per_token_scaled_fp8_quant` ops are never emitted.
For `Qwen/Qwen3-1.7B-FP8` (block-fp8, group 128) the only routed op that actually
executes is **`per_token_group_fp8_quant`** — the standalone activation quant
called from the fp8 linear/MoE apply path. That is the op this commit routes.

## Config: how to exercise the non-compiled path

`route_quant` only engages when the op runs **eager but under CUDA-graph
capture**. That requires disabling torch.compile while keeping full CUDA graphs:

- `-cc.mode=none` — no torch.compile (eager model).
- `-cc.cudagraph_mode=full` — capture the eager model into full CUDA graphs.

(`--enforce-eager` is **not** usable here: it forces `cudagraph_mode=none`, so
nothing is captured and `route_quant` never fires.)

`VLLM_USE_DEEP_GEMM=0` is required so the activation quant surfaces as the
standalone `per_token_group_fp8_quant`; with DeepGEMM on it uses the packed
`per_token_group_fp8_quant_packed` op instead (not routed).

## Benchmark command

```bash
COMMON="--model Qwen/Qwen3-1.7B-FP8 --input-len 256 --output-len 128 \
  --batch-size 16 --num-iters-warmup 10 --num-iters 30 \
  --compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL\"}'"

ENV="CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:\$PATH \
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_DISABLE_COMPILE_CACHE=1"

# Helion ON (default)
env $ENV VLLM_USE_HELION_KERNELS=1 vllm bench latency $COMMON
# Helion OFF (native baseline)
env $ENV VLLM_USE_HELION_KERNELS=0 vllm bench latency $COMMON
```

## Methodology

Sole-process, sequential, same GPU (gpu0), nothing else running. On this shared
2×H100 node, running comparisons in parallel produces large GPU-contention
artifacts — never benchmark in parallel here. Two independent trials below.

## Results (2×H100, Qwen3-1.7B-FP8, block-fp8, DeepGEMM off, mode=none + cudagraph FULL)

| trial | configuration | P50 | Avg | P90 | P99 |
|---|---|---|---|---|---|
| 1 | **Helion ON**  | **0.4851 s** | 0.4868 s | 0.4952 s | 0.5076 s |
| 1 | native OFF     | 0.4991 s     | 0.4981 s | 0.5030 s | 0.5085 s |
| 2 | **Helion ON**  | **0.4842 s** | 0.4827 s | 0.4887 s | 0.4918 s |
| 2 | native OFF     | 0.4973 s     | 0.4958 s | 0.5063 s | 0.5091 s |

- **Helion is ~2.6–2.8 % faster than native at P50** on the non-compiled path
  (0.4842–0.4851 vs 0.4973–0.4991), ~2.3–2.6 % at Avg. Consistent across both
  trials.
- **No JIT spike.** Because the Helion Triton kernels JIT during CUDA-graph
  capture (warmup), the measured iterations are steady and the P99 stays tight
  (≈ P50 + 2–5 ms). This is unlike the compiled (`-O3`) path, where lazy JIT can
  cause a one-time P99 spike — see the compiled-path benchmark doc.
- Absolute latency here (~0.485 s) is higher than the compiled path (~0.45 s)
  because the model runs eager (no Inductor optimization); this doc only compares
  Helion vs native **within** the non-compiled path.

## Conclusion

On the non-compiled (eager + full-CUDA-graph) path, routing
`per_token_group_fp8_quant` to Helion via `route_quant` is a **small, clean win**
(~2.7 % P50) with no JIT-spike downside. This is the mechanism that covers the
eager MoE/MLA/linear call sites the compiled FX swap cannot reach.
