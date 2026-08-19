# FlashInfer NVFP4 native shared-expert integration

## Goal

Use FlashInfer's TRTLLM-gen NVFP4 MoE API added in FlashInfer PR #4239 for
DeepSeek-style models whose routed and shared experts are both serialized
ModelOpt NVFP4. The first validation target is a DeepSeek-R1 NVFP4 v2 checkpoint
on a Blackwell GPU.

This work is stacked on vLLM PR #51754, which provides the generic vLLM
shared-expert representation, appended weight loading, and the distinction
between externally routed and natively fused shared experts.

## Working branch

- Host: `nyx`
- Repository: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/adaptive-prefill-dd10e03-inputs/vllm-dd10e03`
- Branch: `feat/flashinfer-fp4-shared-expert`
- Stack base: `fp4-shared-expert-stack-base` (vLLM PR #51754 commit)

## Execution flow

For each MoE layer, vLLM still computes router logits for only the `E` routed
experts. It stores routed weights followed by `S` shared-expert weights in the
same expert-major tensors. The monolithic FlashInfer call receives:

- `num_experts=E`
- `top_k=K`
- `num_fused_shared_experts=S`
- weight and per-expert scale tensors with `E + S` rows

Inside FlashInfer, the DeepSeek-V3 router chooses `K` routed experts, appends all
`S` shared expert IDs, assigns each shared expert weight `1.0`, and executes the
combined routing, grouped GEMM1, activation, grouped GEMM2, and final reduction.

The model passes `routed_scaling_factor` into FlashInfer. FlashInfer applies it
only to routed contributions; the always-on shared contribution remains at
weight `1.0`. Applying the factor after combining both paths would incorrectly
scale the shared expert.

## Changed files

- `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`
  advertises native support, enforces the FlashInfer contract, disables routing
  replay for this mode, accounts for shared slots in chunk sizing, and passes
  `num_fused_shared_experts` to FlashInfer.
- `vllm/model_executor/models/deepseek_v2.py` enables fusion only for serialized
  ModelOpt NVFP4 weights on SM100-family CUDA when both routed and shared expert
  weights are quantized compatibly and EP/EPLB are disabled. It keeps routed
  scaling inside the kernel. During loading, widened shared `weight` and
  blockwise `weight_scale` tensors are split into appended expert slots, while
  scalar `input_scale` and `weight_scale_2` metadata are replicated.
- `vllm/model_executor/layers/fused_moe/routed_experts.py` includes appended
  shared slots in the global expert count passed to quantization methods. This
  makes globally indexed ModelOpt NVFP4 input-scale storage `E + S` rows rather
  than only `E` rows.
- `vllm/envs.py` exposes
  `VLLM_FLASHINFER_NVFP4_FUSED_SHARED_EXPERTS` (default `1`). Setting it to `0`
  keeps routed experts on the same FlashInfer NVFP4 backend but executes the
  shared expert through the existing separate MLP path. This is the controlled
  A/B switch used for correctness and performance validation.
- `requirements/cuda.txt`, `docker/Dockerfile`, and `docker/versions.json` move
  the Python, cubin, and JIT-cache dependencies to FlashInfer 0.6.17, the first
  release that contains PR #4239.
- `tests/kernels/moe/test_trtllm_nvfp4_shared_experts.py` covers backend
  selection constraints, routing-replay behavior, and `E + S` quantization
  metadata allocation without requiring a GPU.
- `tests/models/test_deepseek_nvfp4_shared_experts.py` covers model-level
  activation and fallback decisions, widened tensor slicing, and scalar NVFP4
  metadata replication.

## Current limitations

- FlashInfer supports this native mode only with `RoutingMethodType.DeepSeekV3`.
- Expert parallelism and EPLB are not supported for native shared experts.
- Routing replay is disabled because its buffer has routed-`K` stride, while
  native fusion internally produces `K + S` assignments.
- Shared-expert weight is fixed at `1.0`.
- Validation requires FlashInfer 0.6.17 and an SM100-family Blackwell GPU.

## Validation completed

- `ruff format` and `ruff check` pass for all changed Python files.
- Python bytecode compilation passes for all changed Python files.
- `git diff --check` passes.
- Container preflight job `2395074` passed 13 focused tests with FlashInfer
  0.6.17 on 8 B200 GPUs.
- Full startup/GSM8K job `2395077` completed with exit code `0:0` using TP=8.
  The server log confirms the FlashInfer monolithic path ran with
  `num_fused_shared_experts=1`. The 64-question smoke score was 95.31%; the full
  1,319-question 5-shot score was 95.45% with 0% invalid responses.
- Exact 8k-input/1k-output A/B job `2395078` completed with exit code `0:0`.
  Each arm used 512 prompts, concurrency 256, infinite request rate, fixed
  lengths, ignored EOS, seed 20260817, and the same TP=8 B200 allocation.

| Metric | Fusion off | Fusion on | Change |
| --- | ---: | ---: | ---: |
| Successful requests | 512 | 512 | no failures |
| Output throughput | 3,480.27 tok/s | 3,556.61 tok/s | +2.19% |
| Total token throughput | 31,322.46 tok/s | 32,009.49 tok/s | +2.19% |
| Mean TTFT | 13,194.19 ms | 12,682.57 ms | -3.88% |
| Mean TPOT | 59.51 ms | 58.43 ms | -1.82% |
| P99 TPOT | 70.27 ms | 68.75 ms | -2.17% |

An InferenceX-style 8k1k concurrency sweep followed in Slurm array job
`2395274`. Each point used 8 B200 GPUs, TP=8, `max-num-seqs=256`, infinite
request rate, ignored EOS, seed 20260817, and ten times the concurrency in
requests. Input lengths were sampled from 6,553 through 8,192 tokens and output
lengths from 819 through 1,024 tokens. Fusion-off and fusion-on used the same
node at each concurrency, with only
`VLLM_FLASHINFER_NVFP4_FUSED_SHARED_EXPERTS` changed.

The current vLLM benchmark client interprets `--random-range-ratio 0.2`
symmetrically around the requested length. To reproduce the older InferenceX
upper-bound convention, the validation harness patched only its node-local
benchmark-client copy so 8,192 and 1,024 remained the upper bounds. It verified
the exact ranges and `num_fused_shared_experts=0` or `1` from each benchmark and
server log before accepting a point.

| Concurrency | Requests | Req/s off | Req/s on | Req/s gain | Output tok/s off | Output tok/s on | Output gain | Mean TTFT reduction | Mean TPOT reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 640 | 2.626 | 2.662 | +1.40% | 2,421.7 | 2,455.5 | +1.40% | +3.67% | +1.29% |
| 128 | 1,280 | 3.297 | 3.385 | +2.67% | 3,045.4 | 3,126.9 | +2.67% | -12.32% | +3.17% |
| 256 | 2,560 | 3.943 | 4.056 | +2.85% | 3,633.8 | 3,737.4 | +2.85% | +0.81% | +2.88% |
| 512 | 5,120 | 3.967 | 4.099 | +3.32% | 3,653.1 | 3,774.4 | +3.32% | +3.20% | +3.29% |
| 1,024 | 10,240 | 3.967 | 4.131 | +4.14% | 3,657.5 | 3,808.9 | +4.14% | +4.01% | +3.94% |

All ten measured arms completed every request with zero failures. Throughput and
mean TPOT improved at every point. Mean TTFT improved at four points but
regressed 12.32% at concurrency 128; p99 TPOT also regressed 5.05% at concurrency
64. Fusion is therefore not a uniformly positive latency result. Because each
arm was measured once and fusion-off always ran first, repetitions with
alternating order are required before treating either regression as a stable
effect. Throughput saturates around concurrency 256 because the server admits at
most 256 active sequences; concurrency 512 and 1,024 primarily increase queue
depth.

Array tasks c64 and c128 are recorded as failed by Slurm because an initial
post-check expected per-request length arrays that the current result JSON no
longer emits. Both measurements themselves completed with zero failures. The
fixed aggregator revalidated request counts, exact sampling-range log messages,
and kernel-mode log messages and generated PASS markers for all five points.

The concurrency-128 point was then repeated in array job `2401083` to determine
whether its 12.32% mean-TTFT regression was stable. Four independent 8x B200
nodes each ran a paired 1,280-request fusion-off/fusion-on comparison. Runs 1
and 3 used off-then-on; runs 2 and 4 used on-then-off. Seeds were 20260819
through 20260822, and each pair reused its node while changing only the fusion
environment variable. All eight measured arms completed every request with
zero failures.

| Repeat | Order | Request throughput gain | Mean TTFT reduction | P99 TTFT reduction | Mean TPOT reduction | P99 TPOT reduction |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | off-on | +2.51% | +3.83% | +4.53% | +2.42% | -1.02% |
| 2 | on-off | +1.76% | +1.47% | -3.65% | +1.75% | -1.85% |
| 3 | off-on | +2.25% | +8.72% | +3.71% | +1.95% | +0.21% |
| 4 | on-off | +2.24% | +7.53% | +10.36% | +2.02% | +1.43% |
| Mean | balanced | +2.19% | +5.39% | +3.74% | +2.04% | -0.31% |

The original mean-TTFT regression did not reproduce: all four repeats improved
mean TTFT. The off-on and on-off mean improvements were 6.27% and 4.50%,
respectively, so the conclusion does not depend on which arm ran first.
Throughput and mean TPOT also improved in all four repeats. P99 TTFT improved in
three of four. P99 TPOT was mixed and averaged a small 0.31% regression, so the
data does not support claiming that fusion improves every tail-latency metric.

A bounded Torch profiler comparison was captured for the same c128 workload.
Each mode warmed up first, skipped ten engine iterations, and captured forty
engine iterations on all eight ranks without stack, shape, or memory recording.
The fusion-off data came from job `2401084`. That job was marked failed only
after its valid trace was written because the harness expected the wrong client
log wording. The corrected fusion-on-only retry, job `2401105`, completed with
exit code zero and wrote into the same result directory.

On rank 0, the trace shows the architectural change directly:

- The separate shared-expert Triton kernel fell from 2,320 calls to zero.
- Standalone FlashInfer FP4 GEMM calls fell from 7,320 to 2,680.
- FP4 conversion kernel calls fell from 12,080 to 7,440.
- Total CUDA kernel calls fell from 110,010 to 91,450 over the same forty engine
  iterations.
- Inclusive CPU annotation time for the 2,320 MoE wrapper invocations fell from
  2,716.95 ms to 1,000.23 ms. The matching profiler summaries report 2,011 ms
  versus 1,912 ms of nested CUDA time for those wrappers.

Summed rank-0 CUDA kernel duration fell 24.13%, but most of that delta came from
the same 4,920 all-reduce calls taking less time in the fusion-on capture.
Because each mode has only one bounded trace, this all-reduce timing difference
is diagnostic and is not attributed to fusion. The removed shared path and the
lower GEMM/conversion call counts are the reliable structural findings; the
balanced end-to-end repeats above are the performance evidence.

A final container preflight in job `2401152` reran all 13 focused tests on B200;
they passed in 14.20 seconds.

The model and result artifacts are stored at:

- Model: `/lustre/fsw/coreai_cserve_dev/kaihangj/models/DeepSeek-R1-NVFP4-v2`
- Startup/GSM8K: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/startup-gsm8k-2395077`
- A/B comparison: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/perf-ab-2395078/comparison.json`
- InferenceX-style sweep summary: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/perf-sweep-2395274/sweep.md`
- InferenceX-style sweep JSON: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/perf-sweep-2395274/sweep_comparison.json`
- c128 repeat summary: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/c128-repeat-2401083/repeats.md`
- c128 repeat JSON: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/c128-repeat-2401083/repeat_comparison.json`
- c128 trace summary: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/c128-trace-2401084/trace_summary.md`
- c128 trace JSON: `/lustre/fsw/coreai_cserve_dev/kaihangj/experiments/flashinfer-r1-nvfp4-v2-validation/results/c128-trace-2401084/trace_summary.json`

## Reproducing the A/B

In the vLLM CUDA development container on a B200, run the selection tests first:

```bash
python -m pytest -q \
  tests/kernels/moe/test_trtllm_nvfp4_shared_experts.py \
  tests/models/test_deepseek_nvfp4_shared_experts.py
```

For the 8k-input/1k-output comparison, launch otherwise identical TP=8 B200
runs with only this environment variable changed:

```bash
# Native shared-expert fusion (default)
VLLM_FLASHINFER_NVFP4_FUSED_SHARED_EXPERTS=1

# Separate shared-expert MLP control
VLLM_FLASHINFER_NVFP4_FUSED_SHARED_EXPERTS=0
```
