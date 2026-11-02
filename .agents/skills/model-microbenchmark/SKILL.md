---
name: model-microbenchmark
description: Time a vLLM model's forward pass on exact batch shapes (decode, prefill, chunked prefill, prefix hit, spec-decode verify, or mixes) with `vllm bench forward`, single-GPU or DP/TP/EP across nodes. Use for per-step model latency, layout/parallelism comparisons, and checking a model-level optimization. Not for single kernels (use kernel-microbenchmark) or end-to-end serving (use `vllm bench serve`).
---

# Model Microbenchmark (`vllm bench forward`)

Builds the batch without the scheduler, runs it through MRV2's real
`execute_model`, and times the forward as a one-shot CUDA graph (prefill
included). The configured drafter is included by default; use
`--exclude-drafter` for target-only timing. All draft tokens are accepted.
Target sampling, rejection sampling, and CPU step overhead are excluded.

## Usage

```bash
vllm bench forward --model deepseek-ai/DeepSeek-V4-Flash \
  --case decode=64*1@8192 --case prefill=8192 --case mix=4096,32*1@8192

# DP/TP/EP: one process per GPU (multi-node: add --nnodes/--node-rank/--master-addr)
torchrun --nproc-per-node=4 -m vllm.entrypoints.cli.main bench forward \
  --model ... -dp 4 --enable-expert-parallel --case decode=128*1@100000
```

Requests: `[N*]Q[@K][c][:D]`, comma-separated, per DP rank.

| Form | Meaning |
| --- | --- |
| `8192` | fresh prefill, 8192 tokens |
| `32*1@8192` | 32 decodes, KV length 8192 (KV length **includes** the query) |
| `4@4096:3` | spec-decode verify, 3 draft tokens |
| `2048@10240` | prefill after an 8192-token prefix-cache hit |
| `2048@10240c` | chunked prefill: the prefix came from earlier chunks |

Verify cases require `--speculative-config` with `num_speculative_tokens`
at least as large as the case's draft count; insufficient capacity is rejected.

Hit vs chunk only differs on models with SWA bounded replay (DSV4.1): a hit
also recomputes the last window (128 tokens) of the prefix.

Or `--shapes file.json|yaml`:
`{"cases": [{"name": ..., "requests": [{"query_len", "kv_len", "num_draft_tokens", "chunked", "count"}]}]}`.

Key flags (plus all `EngineArgs`):

- `--kv-len-jitter 0.5` (default): prefixes drawn within ±50%, same on every
  DP rank, capped at `max_model_len`. Use `0` for exact lengths.
- `--routing auto|real|balanced|zipf`: `auto` = Zipf with dummy weights;
  `zipf` uses fixed alpha=1.25 skew (target only, no PCP or grouped top-k).
- `--iters 20 --samples 5`, `--output-json out.json` (per-rank latency stats,
  `samples_us`, graph-vs-eager check).
- Defaults: `--load-format dummy`, prefix caching off, no torch.compile.

Read `median_us` (each replay is already the max over ranks).

## Kernel breakdown

Add `--profile-dir DIR` to see where the time goes. After timing, each case is
replayed 3 more times under `torch.profiler`; every rank writes
`DIR/<case>.rank<r>.kernels.json` (GPU time per kernel per forward, sorted)
and `DIR/<case>.rank<r>.trace.json.gz` (open in Perfetto). Rank 0's top
kernels are printed and included in `--output-json`.

```bash
python -m vllm.benchmarks.forward.profile DIR/decode.rank0.kernels.json
# Before/after a change, same case: per-kernel deltas, largest first.
python -m vllm.benchmarks.forward.profile base/decode.rank0.kernels.json new/decode.rank0.kernels.json
```

- Kernels are grouped by name without template or function arguments; the
  JSON keeps one full name per group in `example`.
- Only the timed forward is included: the in-graph KV restore and rank
  rendezvous are excluded (`pre_forward_us`). `span_us` (first to last
  forward kernel) should be close to `median_us`, a bit higher from profiler
  overhead; `busy_us` can exceed it when streams overlap.
- Compare the same rank across runs: under EP, ranks differ (e.g. collective
  wait time shows up inside communication kernels).

For Nsight Systems (NVTX, NCCL, CPU side, GPU metrics), skip `--profile-dir`:
the two can't share CUPTI. Each case's 3 profiled replays run between
`cudaProfilerStart`/`Stop` in an NVTX range `forward/<case>`, so a capture range
records only them, one report per case (and rank):

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --cuda-graph-trace=node -o prof vllm bench forward ...       # prof.1, prof.2, ...
torchrun --nproc-per-node=4 --no-python nsys profile ... -o prof_rank%q{RANK} \
  python -m vllm.entrypoints.cli.main bench forward ...
nsys stats --report cuda_gpu_kern_sum --format csv -o kern prof.1.nsys-rep
```

Keep `--capture-range-end=repeat`: nsys's default stops after the first case
and leaves the bench hanging. `nsys stats` also counts the KV restore copies
that the JSON summary excludes.

## Caveats

- **Dummy weights lie for data-dependent kernels.** Dummy gates route every
  token to the same experts (hence synthetic routing). Cross-check suspicious
  kernels with real weights.
- **torch.compile models (e.g. Qwen3/3.5) look slow**: no compile here. Use
  them for correctness only; trust perf for models under `vllm/models/`.
- `sample spread > 0.5%` warns of unstable timing. DSV4.1 DEP prefill is
  inherently ~0.5–1% (MegaMoE cross-rank sync, host engram reads).
- `graph_vs_eager` may be nonzero for small batches where models branch on
  `is_current_stream_capturing()`; that's expected, not a bug.
- KV cache is zero-initialized (`--kv-init zeros`); random KV was within 3%.
- Uniform DP only: every rank gets the same case; no idle ranks.
- CUDA only: it hangs on ROCm (untested cause: likely the in-graph timing
  events).
- Large prefills/prefix hits need room: raise `--max-num-batched-tokens` /
  `--max-model-len` / `--gpu-memory-utilization` if build fails on KV blocks.
- Startup cost is JIT/autotune, not the bench: install FlashInfer jit-cache
  and cubin; DeepGEMM warmup is skipped (each case's eager pass compiles its
  own shapes).
- To compare against serving decode: `vllm bench latency` with 65- vs
  1-token outputs, divided by 64, with a FULL decode graph.
