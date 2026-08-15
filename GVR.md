# GVR for nvidia/GLM-5.2-NVFP4

Explore and implement GVR (<https://arxiv.org/abs/2604.22312v1>) technique for nvidia/GLM-5.2-NVFP4.
Use vllm/models/deepseek_v32/ for the model code (i.e., non-torch-compile version). You will need <https://github.com/vllm-project/vllm/pull/49790> to use this model as it's not default yet.

1. Investigate the overhead of top-k operator in decode. Assume 100K KV length. Benchmark the overhead for different batch sizes (1, 8, 32, 128, 1024) with TP4. Assume no spec decoding for now. Make sure the decode path uses FULL CUDA graph.
2. Investigate the paper and its implementation in TRT-LLM and other projects like SGLang and TokenSpeed. We may reuse some of their code.
3. Implement the GVR technique in vllm/models/deepseek_v32/nvidia/ and benchmark the end-to-end performance with different batch sizes. Measure both TPOT and total TPS. Check whether the measured e2e performance matches the expectation. You will need to use real weights and inputs, since the GVR algorithm is sensitive to the data.
4. Verify the model accuracy with GSM8K test, with real weights.
5. If the performance is not good, investigate the reasons and find a solution to improve the performance.

Document all issues, findings, milestones, and results in this file.
If you add any new CUDA kernels, make sure they can be JIT-compiled for now. Make sure to avoid the full rebuild of vLLM since it takes too long.
Use `VLLM_USE_RUST_FRONTEND=0` to disable the Rust frontend. Python frontend will be enough.

## Working log

Last updated: 2026-08-13. This is a living record; measurements below are kept
even when an approach was rejected.

### Scope and environment

- Checkout: `373592ef57` (detached HEAD). The NVIDIA DeepSeek-V3.2/GLM-5.2
  implementation and routing needed from PR #49790 are already present in this
  checkout.
- Hardware: 4x NVIDIA GB200 (SM100), about 189 GiB per GPU; CUDA 13.0; PyTorch
  2.13.0.
- Model: the locally cached real `nvidia/GLM-5.2-NVFP4` checkpoint (78 layers,
  `index_topk=2048`, `index_topk_freq=4`).
- Required serving configuration: TP4, no speculative decode, FULL CUDA graph,
  and `VLLM_USE_RUST_FRONTEND=0`.
- GVR is opt-in with `VLLM_USE_GVR_TOPK=1`. Unsupported configurations fail
  closed: non-SM100, missing CuTe DSL, `index_topk != 2048`, speculative decode,
  PP, DCP, or PCP.
- Duplicate-work check: vLLM draft PR #44606 changes sparse-top-k dispatch
  thresholds and adds a benchmark, but does not implement temporal GVR. No open
  vLLM PR implementing GVR was found. This work must not be proposed as a PR
  without repeating the mandatory issue/PR checks in `AGENTS.md`.

The component timings in the first pass use one GB200 because tensor parallelism
does not shard this replicated selector. Step 1 specifically requires the
percentage of a real TP4 100K decode forward consumed by all top-k invocations,
not only isolated operator latency. For each requested batch, the final report
will therefore include decode-step latency, summed top-k CUDA time across the
model's indexer layers, top-k percentage, and the corresponding ideal speedup
ceiling. The isolated timings below remain supporting diagnostics.

### Algorithm and implementation research

The [GVR paper](https://arxiv.org/abs/2604.22312) exploits the temporal coherence
of sparse-attention logits. It uses the preceding decode step's top-k indices as
a value sample, guesses a threshold, verifies the number of current logits above
that threshold, and exactly refines/collects when the candidate count is outside
the accepted range. It is an exact selector rather than an approximate top-k.
The paper's production configuration uses K=2048, a candidate capacity of 6144,
512 threads, about 60 KiB of shared memory, and one CTA per row. Against its
production radix baseline it reports 1.88x average operator speedup (up to 2.42x)
and up to 7.52% TPOT improvement at 100K context without MTP. Its GSM8K result is
95.23 with GVR versus 95.11 for the baseline.

NVIDIA merged the original implementation in
[TensorRT-LLM PR #12385](https://github.com/NVIDIA/TensorRT-LLM/pull/12385).
TensorRT-LLM has since evolved it into a family of CuTe DSL kernels with
single-/multi-CTA, register-resident, and throughput-oriented tiers. The current
sources inspected were
[`gvr_topk_decode.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py),
`gvr_topk_decode_reg.py`, `gvr_topk_decode_tp.py`, and their dispatcher at commit
`3c68ae6ac79c48c6bad5816adcfcefc7d9897d55`. The implementation here adapts the
exact multi-CTA CuTe DSL kernel and its block-scan dependency. These kernels JIT
compile; no vLLM C++/CUDA rebuild is required.

Repository code searches for `GVR`, `Guess-Verify-Refine`, and related names in
[SGLang](https://github.com/sgl-project/sglang) and
[TokenSpeed](https://github.com/ByteDance-Seed/TokenSpeed) found no GVR kernel as
of 2026-08-13. SGLang contains other sparse-indexer optimizations, including
index-cache work, but no reusable GVR implementation was identified.

### Baseline and rejected implementations

vLLM's current baseline is newer than the radix selector used by the paper: it
uses `cooperative_topk` through batch 32 on SM90+ and `persistent_topk` for larger
batches. That substantially changes the break-even point.

The first implementation adapted the original TensorRT-LLM CUDA GVR kernel as a
PyTorch JIT extension. It was exact on temporally correlated synthetic data, but
lost at every requested batch size and was removed:

| Batch | Existing vLLM | Original CUDA GVR | Existing / GVR |
|---:|---:|---:|---:|
| 1 | 25.696 us | 67.984 us | 0.378x |
| 8 | 26.496 us | 57.536 us | 0.461x |
| 32 | 28.032 us | 67.984 us | 0.412x |
| 128 | 41.344 us | 71.584 us | 0.578x |
| 1024 | 266.704 us | 536.512 us | 0.497x |

The current TensorRT-LLM CuTe DSL tiers were then evaluated at a 100,032-element
padded stride with 100,000 valid candidates. Kernel-only experiments showed:

- Register-resident GVR: 20.960 us at batch 1 and 21.344 us at batch 8, but
  42.912 us at batch 32. This looked attractive until request-index gather and
  state-scatter launches were included.
- Base multi-CTA GVR: 36.704 us at batch 128 and 202.768 us at batch 1024,
  versus 40.832 us and 267.136 us for `persistent_topk`.
- The throughput-specific GVR tier was slower in this environment: 77.696 us at
  batch 128 and 430.800 us at batch 1024.

The full temporal-state wrapper (hint gather + exact selector + state scatter)
made small-batch GVR unprofitable and also made baseline fallback rows slower if
their state was refreshed every step:

| Batch | Existing selector | Full GVR or fallback+state | Ratio |
|---:|---:|---:|---:|
| 1 | 26.343 us | 75.868 us GVR | 0.347x |
| 8 | 26.532 us | 73.784 us GVR | 0.360x |
| 32 | 26.665 us | 46.140 us fallback+state | 0.578x |
| 128 | 41.332 us | 53.402 us fallback+state | 0.774x |
| 1024 | 266.275 us | 236.819 us GVR | 1.124x |

The cause is launch overhead: the register GVR kernel itself is fast, but the
two Triton launches needed to gather/scatter stable per-request state erase that
win at small batch. Stable request identity cannot safely be replaced by batch
row identity because continuous batching reorders rows.

### Pre-rewrite implementation and dispatch decision

Before the structural rewrite, the guarded hybrid used GVR only where the
complete path, including state management, beat vLLM's selector. A threshold
sweep at 100K produced:

| Batch | `persistent_topk` | Full multi-CTA GVR | Speedup |
|---:|---:|---:|---:|
| 256 | 76.18 us | 80.88 us | 0.942x |
| 384 | 115.12 us | 122.85 us | 0.937x |
| 512 | 151.72 us | 135.33 us | 1.121x |
| 768 | 222.89 us | 185.90 us | 1.199x |
| 1024 | 264.47 us | 228.84 us | 1.156x |

Consequently, the pre-rewrite dispatch required batch >=512, K=2048, and a row
stride between 65,536 and 262,144 divisible by 64. The structural rewrite below
supersedes this threshold.

Temporal state is per model layer and keyed by vLLM's stable scheduler request
slot. Metadata pads nonexistent FULL-CUDA-graph rows with request index `-1`, so
they cannot corrupt live state. New/cold requests use evenly spaced indices as a
safe initial hint. The implementation adds:

- CuTe DSL GVR and block-scan code under
  `vllm/models/deepseek_v32/nvidia/ops/`;
- small Triton kernels for stable-request hint gather, decode-state scatter,
  and prefill-state seeding;
- stable request indices in common/indexer attention metadata;
- per-layer GVR state allocation in the NVIDIA GLM/DeepSeek-V3.2 model; and
- opt-in selector dispatch in `sparse_attn_indexer`.

### Correctness and validation progress

Before writing tests, the test contract was defined as follows: the module maps
float32 per-row logits plus causal sequence lengths and a temporal top-k hint to
exact int32 top-k indices; the failure risks are an inexact cold start and hints
following batch rows instead of stable requests; focused CUDA unit tests are the
cheapest level that catches both, while real-model GSM8K covers behavioral
accuracy.

Completed checks:

- Standalone exactness against `torch.topk` passed for cold-start batches 1, 8,
  256, and 512 (the committed unit uses batch 512, where dispatch is enabled).
- Stable-request reorder/state-scatter unit test passed, including a `-1`
  padding request.
- Command:
  `CUDA_VISIBLE_DEVICES=0 VLLM_USE_RUST_FRONTEND=0 .venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k 'gvr_' -v`
  — 2 passed, 173 deselected.
- Python compilation and import/custom-op schema registration passed.
- Hand-written changed files pass Ruff. The two Apache-licensed adapted
  TensorRT-LLM source files carry their upstream commit in a source note and a
  file-level Ruff exemption; the repository formatter was applied in the final
  validation pass.
- A TP4 real-weight smoke run completed successfully with
  `VLLM_USE_GVR_TOPK=1`, `VLLM_USE_RUST_FRONTEND=0`, no spec decode, a 100,032
  max model length, and a requested FULL graph. The sparse MLA backend resolved
  this to `FULL_AND_PIECEWISE`: prefill was captured piecewise and decode was
  explicitly captured as FULL. All four ranks completed graph capture, and the
  real checkpoint generated `" yes,"` for the smoke prompt.
- A direct batch-512 GVR warmup, `torch.cuda.CUDAGraph` capture, and replay at a
  100,032 padded stride also passed. This separately confirms that the CuTe DSL
  multi-CTA launch and the Triton gather/scatter launches are graph-capturable
  on SM100; the batch-1 model smoke used the baseline selector by design.
- The 432.90 GiB checkpoint took 220.09 seconds to read and 299-301 seconds for
  each rank to finish model loading. Each rank reported 106.64 GiB for model
  loading and 107.83 GiB total model/non-torch usage. Initial engine setup took
  404.02 seconds, including 104.52 seconds of compilation. The first run also
  spent about two minutes populating FlashInfer's persistent NVFP4 MoE autotune
  cache; later runs can reuse it.
- NVFP4 autotuning emitted many recoverable 20 MiB allocation-failure warnings
  while probing configurations. They were not a fatal model OOM: loading,
  warmup, graph capture, and inference all completed. The engine retained about
  60.8 GiB per GPU for KV cache, corresponding to 1,368,512 unshared tokens or
  13.68 independent 100,032-token requests. Batch 1024 at 100K therefore must
  use one physically shared prefix through prefix caching.
- The installed `deep_ep` and `deep_gemm` wheels have a PyTorch ABI mismatch.
  vLLM logged the import failures and successfully selected its vendored
  DeepGEMM fallback; no environment mutation or vLLM rebuild was needed.
- The server originally labeled the real-weight GVR evaluation reserved 20 GiB
  per rank for KV cache,
  exposed 450,176 tokens of cache, and captured PIECEWISE and explicit FULL
  decode graphs at batches 1, 8, 32, 128, 512, and 1024. Its engine setup took
  254.65 seconds after model loading, including 106.31 seconds of compilation.
  The dispatch audit later proved this was the legacy model path, not GVR.

The first corrected 100K-prefix run generated exactly 32 tokens per branch. It
was originally labeled GVR, but later kernel-level trace validation proved that
the server resolved GLM-5.2 to the legacy `deepseek_v2` model class and GVR
never dispatched. These serving numbers are retained as scheduler/admission
diagnostics only. One request with `n=batch` makes the branches share the
physical prefix; branches are admitted progressively, so aggregate throughput
is not a fixed-batch model-forward latency.

| Batch | Generated tokens | Request wall time | Aggregate throughput |
|---:|---:|---:|---:|
| 1 | 32 | 0.491 s | 65.22 tok/s |
| 8 | 256 | 1.047 s | 244.40 tok/s |
| 32 | 1,024 | 2.356 s | 434.55 tok/s |
| 128 | 4,096 | 6.585 s | 621.99 tok/s |
| 1024 | 32,768 | 47.172 s | 694.65 tok/s |

The otherwise matching legacy baseline request-level pass measured 61.95, 131.40,
303.49, 606.86, and 689.76 tok/s for batches 1, 8, 32, 128, and 1024. These are
not sustained fixed-batch throughputs. A single `n=batch` request progressively
admits child sequences. At batch 1024, the trace shows many mixed steps with
35-62 context requests (roughly 1.1K-2.0K context tokens) while the decode batch
ramps through 67, 128, 188, 247, and so on. Only seven per-rank steps reached a
pure batch-1024 decode plateau. The 100,000-token prompt leaves a 32-token tail
after 64-token prefix-cache blocks, so every admitted child also has context
work under the 2,048-token scheduler budget. Consequently, the 689.76 and
694.65 tok/s figures include this long admission/prefill ramp and must not be
reported as batch-1024 e2e decode throughput or as a measured GVR speedup.

CUDA graphs were enabled: both legacy-class servers explicitly captured FULL
decode graphs at batch 1024. The baseline's pure batch-1024 FULL-graph forward
averaged 87.755 ms, corresponding to 11,668 decode tok/s. This is not the final
portable-model baseline; see the corrected controlled comparison below.

The first follow-up run cannot be counted as a GVR result. Its pure batch-1024
forward averaged 87.192 ms versus 87.755 ms for the baseline (a nominal
1.00645x), but the GPU trace contained the baseline
`FilteredTopKUnifiedKernel` and no GVR hint, selector, or state-update kernels.
The 0.64% latency difference is therefore ordinary inter-run variation, not a
measured GVR speedup. The report is `prof/gvr_enabled.nsys-rep`.

Two independent integration requirements were missing. First, the initial
server selected the default `vllm.v1.worker.gpu_model_runner.GPUModelRunner`;
its construction of `CommonAttentionMetadata` does not populate the new stable
`request_indices` field. Second, the checkpoint architecture
`GlmMoeDsaForCausalLM` still resolves to the legacy `deepseek_v2` class unless
the development-only model-class override from PR #49790 is supplied. The
legacy class never allocates the GVR state tensors at all. A second attempted
run with only `VLLM_USE_V2_MODEL_RUNNER=1` averaged 84.464 ms, but its trace
again contained only `FilteredTopKUnifiedKernel`; it is also invalid as a GVR
measurement. Its report is `prof/gvr_v2_enabled.nsys-rep`.

The valid controlled pair must set both `VLLM_USE_V2_MODEL_RUNNER=1` and
`--model-class-overrides '{"GlmMoeDsaForCausalLM":
"vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}'`. Both sides
must use those settings so the comparison does not mix model classes or runner
overheads.

### Corrected real batch-1024 e2e speedup

The controlled TP4 comparison is complete. Both runs used the portable model
class, V2 GPU runner, 100,032 max model length, no speculative decoding, the
same 20 GiB KV cache per rank, and explicit FULL decode graphs at batch 1024.
Only `VLLM_USE_GVR_TOPK` differed. Each mean contains the same seven steady
decode steps on each of four ranks (28 rank-steps total).

| Selector | Model forward | Fixed-batch throughput | Relative result |
|---|---:|---:|---:|
| Baseline persistent top-k | 66.542 ms | 15,389 tok/s | 1.000x |
| GVR | 62.161 ms | 16,473 tok/s | **1.0705x** |

GVR reduces complete model-forward latency by **6.585%** and raises fixed-batch
decode throughput by **7.049%**. Treating each TP step's slowest rank as the
critical path gives a corroborating 66.576 ms versus 62.254 ms, or 1.0694x.
Excluding the first plateau transition step gives 1.071x, so the result is
stable around **1.07x** rather than being driven by that transition.

The kernel attribution explains the result. Baseline top-k consumes 6.270 ms
per rank-step, or 9.42% of the portable model forward; its zero-cost ceiling is
1.104x. GVR's complete replacement cost is 1.986 ms: 1.729 ms for the CuTe
selector, 0.145 ms to prepare hints, and 0.112 ms to store state. The measured
4.283 ms selector saving closely matches the 4.382 ms end-to-end saving.

Trace validation passed. The GVR plateau has exactly 147 launches per rank
(`7 steps * 21 indexer layers`) of each GVR selector, hint, and state-store
kernel and no `FilteredTopKUnifiedKernel`. The matching baseline has exactly
147 filtered top-k launches per rank and no GVR kernels. Reports are
`prof/gvr_real_enabled.nsys-rep` and
`prof/gvr_real_baseline.nsys-rep`.

Before the structural rewrite, the hybrid did not dispatch GVR for the
originally requested smaller batches (1, 8, 32, and 128) because its threshold
was 512 rows. The later rewrite supersedes that dispatch result.

### Forced-GVR small-batch e2e result (no fallback)

The requested no-fallback comparison is complete. The GVR dispatch threshold
was temporarily lowered from 512 to one row, then restored after profiling.
Both sides used the same portable model, V2 GPU runner, TP4 configuration,
100,032-token context, and FULL CUDA graphs. The table reports the median
steady decode forward on the TP critical path: for each step, take the slowest
of the four ranks, then take the median over 27--31 steps. This robust statistic
excludes isolated first-use/JIT stalls visible in both traces.

| Batch | Baseline forward | Forced GVR forward | Baseline TPS | GVR TPS | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.644 ms | 7.798 ms | 130.8 | 128.2 | **0.980x** (-2.0%) |
| 8 | 8.416 ms | 8.551 ms | 950.6 | 935.5 | **0.984x** (-1.6%) |
| 32 | 10.395 ms | 10.416 ms | 3,078 | 3,072 | **0.998x** (-0.2%) |
| 128 | 15.698 ms | 15.199 ms | 8,154 | 8,422 | **1.033x** (+3.3%) |

This is genuine forced GVR, not the production fallback path. At batches 1,
8, 32, and 128, respectively, each rank launched 651, 609, 609, and 567 GVR
kernels, with matching counts for hint preparation and state storage, and zero
baseline selector kernels. Those counts are exactly `decode steps * 21
indexer layers`. The matched baseline instead used cooperative top-k at batches
1, 8, and 32 and filtered top-k at batch 128, with no GVR kernels.

The selector attribution agrees with the full-model crossover. Baseline versus
GVR replacement cost per rank-step was 0.284 versus 0.398 ms at batch 1,
0.286 versus 0.384 ms at batch 8, 0.361 versus 0.404 ms at batch 32, and 0.914
versus 0.431 ms at batch 128. Thus forcing GVR loses to the specialized
cooperative selector at very small batches, reaches parity near batch 32, and
wins once the filtered baseline becomes expensive. Baseline selector share of
the complete forward was 3.71%, 3.40%, 3.47%, and 5.82%, giving theoretical
zero-cost ceilings of 1.039x, 1.035x, 1.036x, and 1.062x.

Reports are `prof/gvr_forced_small_baseline.nsys-rep` and
`prof/gvr_forced_small.nsys-rep`. At the time of these traces, the checked-in
production dispatch still started at batch 512; the structural rewrite below
later changes the crossover.

### Small-batch optimization follow-up

The forced-GVR traces split the 21-layer per-step replacement cost into the
selector, hint preparation, and state storage. At batches 1, 8, 32, and 128,
the respective totals were 0.314/0.053/0.031 ms,
0.296/0.054/0.034 ms, 0.312/0.056/0.036 ms, and
0.333/0.059/0.039 ms. The two state-management kernels therefore consume
0.084--0.098 ms per model forward and are the first optimization target.
Removing them as separate launches requires preserving stable request identity
inside the GVR kernel: load hints indirectly from per-request state and write
the new state while emitting compact output. Launch-shape tuning and the
register-resident TensorRT-LLM tier are secondary candidates.

The full-model data also bounds what fusion can accomplish. If hint preparation
and state storage became literally free, forced-GVR forward latency would be
7.714, 8.463, 10.324, and 15.101 ms at batches 1, 8, 32, and 128. Relative to
the measured baseline, those optimistic upper bounds are 0.991x, 0.994x,
1.007x, and 1.040x. Therefore state fusion alone cannot make batches 1 or 8
win; batch 32 has less than 0.7% available, while batch 128 is the first useful
target.

Two lower-risk tuning checks were rejected as primary solutions:

- Changing the Triton gather/scatter block from 256 threads to 128, 512, or
  1024 produced only sub-microsecond ordering changes under CUDA-graph replay,
  with no consistent winner across batches. These helpers are already close to
  the graph-launch floor.
- A cluster-size sweep of the CuTe selector on perturbed synthetic logits did
  not reproduce real-model kernel timing or rank the current launch policy
  reliably. GVR admission cost depends on the real logit/hint distribution, so
  synthetic launch-shape results must not drive the production dispatch.

The upstream register-resident tier loads each row once and keeps it in
registers, but the already-recorded 100K measurements were 20.960 us at batch 1
and 21.344 us at batch 8. The real-model selector in the forced traces averages
14--16 us per layer, so importing that tier does not address the current
bottleneck.

Recommended order is therefore: (1) measure real batches 64, 128, 256, 384,
and 512 to define the non-monotonic dispatch regions; (2) enable the existing
path at any verified winning tier, with batch 128 already showing 1.033x e2e;
(3) implement a fused stable-state CuTe entry that reads per-request hints and
dual-writes compact output plus persistent state; and (4) remeasure before
considering deeper selector changes. Keep cooperative top-k for batches 1--32:
it is already the better small-grid algorithm and avoids all temporal-state
traffic.

### Can the GVR selector itself improve drastically?

Kernel speedup and model speedup have different ceilings. The complete current
GVR replacement (selector plus state helpers) is only 5.10%, 4.49%, 3.88%,
2.83%, and 3.20% of the forced-GVR forward at batches 1, 8, 32, 128, and 1024.
Even making that entire path free would improve the current GVR forward by only
1.054x, 1.047x, 1.040x, 1.029x, and 1.033x. Relative to the original baseline,
the corresponding absolute ceilings are 1.033x, 1.031x, 1.038x, 1.063x, and
1.106x. A dramatic e2e improvement is therefore impossible from the selector
alone.

There is nevertheless one credible large *kernel* optimization. The current
R0 path gathers hint scores, builds histogram rungs, scans the complete row to
count several thresholds, chooses an admitted threshold, and then scans the row
again to collect candidates before exact Phase 4. It avoids an extra count
rescan by caching per-thread counts, but it cannot reuse the values from that
count scan. A speculative verify-and-collect fast path could instead:

1. derive one likely threshold from the temporal hints;
2. perform one row scan that both counts and bounded-appends candidates;
3. run the existing exact Phase 4 when the count is in `[K, kC]`; and
4. fall back to the current multi-rung/refine path on underflow or overflow.

This remains exact because an unsuccessful speculation discards its candidates
and takes the existing path. A successful row removes one full logits scan, so
roughly 1.5--2x selector speedup is plausible if real admission hit rate is
high. It is not yet a measured result. The first experiment should add
development-only per-row counters for selected rung, candidate count, R0 miss,
and number of full scans, then capture them on real decode data. Implement the
fast path only if those counters show a high one-shot success rate.

The only route to materially more e2e headroom is producer fusion. The
DeepGEMM paged-MQA logits kernels consume approximately 0.156, 0.308, 0.754,
2.617, and 19.601 ms per model step at batches 1, 8, 32, 128, and 1024, much
more than GVR itself. Computing the temporal threshold first and folding
bounded candidate collection into the logits epilogue could avoid materializing
and rereading the full `[batch, 100K]` fp32 logits tensor. That is a DeepGEMM
and indexer redesign rather than an isolated GVR optimization, has a much
larger correctness/fallback surface, and conflicts with the current preference
for JIT-only kernels unless implemented as a separate CuTe MQA kernel.

Other ideas are lower priority: storing the previous numeric cutoff can skip
the 2K hint histogram but is less robust to score-scale drift; the
register-resident tier already lost on measured shapes; and approximate
cross-layer/top-k reuse would increase headroom only by giving up the exactness
property validated by the current GVR path.

The benchmark client initially undercounted generated tokens because vLLM can
suppress empty decoded fragments and later stream several token IDs together.
It was corrected to request and sum `token_ids`; the table above contains only
the corrected run. `min_tokens=max_tokens`, `ignore_eos=true`, and a fixed seed
kept every branch alive for the complete measurement window.

The first `bash run_gsm8k.sh gsm8k_gvr` attempt stopped before inference:
`datasets` could not create its lock in the shared, read-only
`/mnt/lustre/hf-models/datasets` cache. The rerun redirects only
`HF_DATASETS_CACHE` to a writable temporary directory.

The run originally labeled the corrected GVR evaluation completed all 1,319
GSM8K examples in 2:55 of API request time. With five shots, lm-eval task
version 3, and the full dataset, both `flexible-extract` and `strict-match`
exact match were **0.9363 ± 0.0067**.
The command was
`HF_DATASETS_CACHE=/tmp/gvr-hf-datasets bash run_gsm8k.sh gsm8k_gvr`; aggregated
results and samples were written under `gsm8k_gvr/`.

The matching legacy baseline completed all 1,319 requests in 2:50. Its exact
match was
**0.9356 ± 0.0068** with `flexible-extract` and **0.9348 ± 0.0068** with
`strict-match`. The nominal differences were +0.0008 and +0.0015 respectively
before display rounding, both far below one standard error. The command was
`HF_DATASETS_CACHE=/tmp/gvr-hf-datasets bash run_gsm8k.sh gsm8k_baseline`, and
results were written under `gsm8k_baseline/`. The later dispatch audit proved
that both evaluations used the legacy selector, so their near-equal scores do
not validate real GVR accuracy. A portable-model GSM8K comparison with the
explicit class override remains required.

### Step 1: top-k share of end-to-end model forward

The ceiling measurement is complete. The baseline TP4 server was run under
Nsight Systems 2025.5.2 with `--cuda-graph-trace=node` and dynamic CUDA profiler
ranges. Each worker step carries a vLLM NVTX label such as
`execute_context_0(0)_generation_128(128)`. Only the exact steady-decode label
for each requested batch was retained. The denominator is Nsight's GPU
projection of that complete `execute_model` range; the numerator is the summed
CUDA duration of the 21 sparse-indexer top-k invocations attributed to the same
range. Values below are averages over all retained rank-steps on four TP ranks.

| Decode batch | Rank-steps | Model forward | Summed top-k | Forward share | Ideal ceiling |
|---:|---:|---:|---:|---:|---:|
| 1 | 124 | 9.682 ms | 0.301 ms | 3.11% | 1.032x |
| 8 | 116 | 11.957 ms | 0.300 ms | 2.51% | 1.026x |
| 32 | 116 | 16.994 ms | 0.405 ms | 2.38% | 1.024x |
| 128 | 104 | 26.802 ms | 0.977 ms | 3.64% | 1.038x |
| 1024 | 28 | 87.755 ms | 5.651 ms | 6.44% | 1.069x |

The ideal ceiling is Amdahl's-law `1 / (1 - top-k share)`: it assumes top-k can
be removed entirely with no replacement cost and that all of its summed CUDA
time lies on the critical path. It is therefore deliberately optimistic. The
real ceiling peaks at only about 6.9% speedup for batch 1024; smaller batches
offer roughly 2.4-3.8% even under the zero-cost assumption.

This original table used the legacy checkpoint registry target. With the
required portable-model class override, the corrected batch-1024 baseline
spends 6.270 ms of a 66.542 ms forward in top-k: **9.42%**, for a zero-cost
ceiling of **1.104x**. The other batch sizes have not yet been remeasured with
the portable class.

Trace consistency checks passed. Batch 1 contained 124 rank-step ranges and
2,604 selector kernels (`124 * 21`); the other batches likewise contained 21
selector kernels per rank-step. The actual real-model baseline dispatch was
`cooperative_topk` for batches 1, 8, and 32, and
`FilteredTopKUnifiedKernel` for batches 128 and 1024. This is more authoritative
than the isolated synthetic selector labels used during early kernel screening.
The reports are `prof/gvr_baseline.2.nsys-rep` through
`prof/gvr_baseline.6.nsys-rep`; report 1 is the empty rejected 64-output-token
attempt and is excluded.

### Final validation

The two GVR CUDA tests passed again after formatting; 44 neighboring
short-prefill, metadata, slot-mapping, and DCP tests passed; Ruff check and Ruff
format hooks passed on all changed Python files; targeted `compileall` and
`git diff --check` passed. The real batch-1024 performance measurement is now
complete. Portable-model GSM8K and the smaller-batch portable-model profiles
remain to replace the legacy-class measurements identified by the dispatch
audit.

### Exact algorithm rewrite exploration (in progress)

The next optimization pass is using captured GLM-5.2-NVFP4 routing logits,
rather than synthetic random tensors, to evaluate several exact speculative
algorithms. The capture points are the input logits and prepared temporal hints
immediately around the GVR launch. Candidate admission is always checked against
the exact `[K, C] = [2048, 6144]` safety window; an underflow or overflow must
fall back to an exact selector, so none of these experiments relax correctness.

The first attempted batch capture used `n=N` completions from one request. It
did not create the intended decode workload: batch 128 consisted of short
length-3 fork rows, while only one row retained the 100K context. Those tensors
were rejected and moved to `/tmp/gvr_invalid_captures_20260813_1706`. The
corrected harness sends independent concurrent requests against a shared cached
100K prefix. A capture-time assertion now rejects the entire tensor unless its
minimum sequence length is at least 65,536. The production GVR size guard is
lowered only in the temporary capture build and will be restored before any
candidate is retained.

The approaches being compared are:

1. A one-pass speculative collect from a temporal-hint quantile. It replaces
   the current full-row admission count followed by a second full-row collect
   with one bounded append pass; exact fallback handles counts outside
   `[K, C]`.
2. The minimum temporal-hint score. With 2,048 unique valid hints it guarantees
   at least K candidates, but can overflow C; this tests whether a particularly
   simple proof-driven threshold is selective enough in practice.
3. The previous step's numeric exact cutoff. It avoids the 2K hint-score
   histogram entirely, at the cost of sensitivity to layer-local score drift.
4. GLM-specific rung ladders and launch/cluster configurations in the existing
   exact kernel, measured on the same captured logits.
5. Producer fusion, which would collect candidates in the paged-MQA logits
   epilogue and avoid materializing and rereading `[batch, 100K]` fp32 logits.
   This has the largest theoretical upside but is also the largest redesign.

`benchmarks/kernels/benchmark_gvr_thresholds.py` now reports temporal overlap,
candidate-count distributions, admission/underflow/overflow rates for hint
quantiles, and consecutive-step numeric-cutoff results. Final numbers are
recorded below from the corrected capture run.

The corrected capture run produced 246 finite long-context rows: batch 1, 8,
and 32, sampled at three layers over two consecutive decode steps. All valid
row lengths were 100,001-100,004. A 128-request client wave was scheduler-split
into 32-row model batches under the configured 20 GiB KV-cache budget, so no
batch-128 capture is claimed.

The threshold census produced these aggregate results:

| Speculative threshold | Fast-path admission | Underflow | Overflow |
|---|---:|---:|---:|
| minimum hint (`q=.00`) | 72.4% | 0.0% | 27.6% |
| hint `q=.01` | 78.5% | 0.0% observed | 21.5% |
| hint `q=.05` | 74.8% | 3.7% | 21.5% |
| hint `q=.10` | 48.0% | 33.3% | 18.7% |
| hint `q=.15` | 58.1% | 38.2% | 3.7% |
| hint `q=.35` | 21.5% | 78.5% | 0.0% |

The minimum-hint proof is valid but not selective enough: candidate count was
10,935 on average and reached 60,625, versus capacity 6,144. Previous numeric
cutoff reuse was also unstable: only 3.1-50.0% of multirow samples landed in
the safety window, depending on layer and batch. Both ideas are rejected as
standalone fast paths.

A GLM-calibrated rung set `(q=.35, .05, .01)` covered 100% of captured rows,
where the existing `.60/.35` hint pair alone covered 21.5%. Silicon timing
nevertheless rejected it. Value-exact CUDA-graph replay means over all six
layer/step captures was:

| Batch | Current default | GLM 3 rungs | GLM 3 + virtual seed | Secant |
|---:|---:|---:|---:|---:|
| 1 | 16.985 us | 21.156 us | 19.643 us | 18.913 us |
| 8 | 19.418 us | 22.048 us | 20.088 us | 18.963 us |
| 32 | 19.759 us | 22.727 us | 20.654 us | 19.589 us |

Thus better first-pass admission does not imply a faster kernel: the wider
multi-count phase and altered candidate distributions cost more than the
fallbacks they remove. Cluster-size, SMEM-cache, and read-only-load sweeps also
failed to improve the default. Cluster 1 was 14-22% slower, cluster 2 was
9-10% slower, cluster 8 was 2% slower at batches 1/8 and 2.6x slower at batch
32, SMEM caching was 2-5% slower, and read-only loads were neutral.

One comparison exposed a separate baseline-selector issue. On
`gvr_b32_call20.pt`, `cooperative_topk` differed from `torch.topk` on seven
rows, by as many as 125 of 2,048 values and 0.0257 score units. GVR matched
`torch.topk` exactly on every row. Therefore the cooperative selector's lower
latency cannot be treated as an exact-algorithm target for this study.

The exactness audit was then expanded to all 246 captured rows. GVR had zero
value-set mismatches against `torch.topk`; `cooperative_topk` mismatched eight
rows in two captures. This is not a tie-order artifact because the audit sorts
the selected fp32 score values before comparison.

One phase-level option did help at small batch: disabling fused
rank-and-scatter and using the older exact histogram-snap Phase 4 improved the
main kernel by 1.008x, 1.037x, and 1.036x at batches 1, 8, and 32. Combining it
with 128-bit loads was neutral, while SMEM caching and 1,024 threads lost. The
win does not scale to the production batches: repeating captured real rows to
isolate occupancy gave 0.943-0.959x at batch 128 and 0.932-0.940x at batch
1024. It is therefore a possible small-batch specialization, not a replacement
for the current production Phase 4. Since the production guard remains batch
512, it is not integrated into dispatch yet.

Applying the measured Phase-4 gain to the prior forced-GVR e2e profiles gives
only a modeled 0.003, 0.011, and 0.011 ms reduction per model step at batches
1, 8, and 32. The corresponding baseline/GVR ratios remain approximately
0.981x, 0.985x, and 0.999x. In other words, even the best exact kernel-local
variant does not turn forced small-batch GVR into a meaningful real e2e win.
An additional server run would be dominated by normal run-to-run noise at this
scale, so the losing variant was not integrated merely to produce that run.

The outcome of this search is that local retuning cannot produce a drastic
gain. The best exact small-batch phase substitution is about 3.6-3.7%, and all
cluster/count/rung variants were neutral or slower. A larger gain requires the
structural one-pass path described above (speculative bounded collect followed
by exact fallback) or fusion with the MQA logits producer. The captured data
also shows why the fallback is essential: no single hint threshold achieved
both zero underflow and zero overflow across layers.

### Structural rewrite: fused temporal state

The structural pass is complete. It explored the proposed one-pass algorithm
before changing the production dataflow. Four exact speculative collectors used
the minimum temporal-hint threshold, accepted only candidate counts in
`[K, C]`, and otherwise fell back to the unchanged exact GVR path:

- one shared atomic append stream ran at only 0.59-0.66x the current kernel;
- per-thread register caches followed by shared compaction ran at 0.61-0.67x;
- a scalar warp-segmented collector ran at about 0.61x; and
- vectorizing the warp-segmented collector reduced it further to about 0.36x.

All variants were value-exact, but ballot/compaction, shared atomics, and
cluster handoffs cost more than the full-row reread they eliminated. These
experimental paths were removed. A transient apparent 2x regression after the
revert was traced to an orphaned rejected benchmark holding GPU 0 at 100%; after
stopping that exact process, the original real-capture baselines returned to
16.982, 19.430, and 19.765 us at batches 1, 8, and 32.

The winning rewrite instead eliminates the two wrapper kernels. The CuTe GVR
CTA now:

1. maps each decode row to its stable scheduler request slot;
2. reads the persistent previous top-k row directly during Phase 1 and Phase
   1b, with no intermediate hint tensor;
3. synthesizes the same evenly spaced cold-start indices inline when the state
   is invalid;
4. emits the compact top-k output; and
5. persists that output with aligned 128-bit copies and marks the request state
   valid in the same launch.

The initial fused prototype staged all 2,048 hints through shared memory. It was
exact but slower because the added copy and barrier outweighed the removed
launch. Direct zero-copy reads fixed that, and replacing four scalar state-copy
iterations per thread with one 128-bit copy produced the final result. The
per-model hint buffer and its custom-op plumbing were removed. Prefill state
seeding remains separate because it runs on a different path and frequency.

CUDA-graph replay on all six real 100K layer/step captures gives the following
complete-path result. `Current` includes Triton hint preparation, the CuTe GVR
selector, and Triton state storage. `Fused` includes direct hint access, the
same exact selector, and in-kernel state persistence.

| Batch | Current | Fused | Pipeline speedup |
|---:|---:|---:|---:|
| 1 | 21.533 us | 19.575 us | **1.100x** |
| 8 | 22.817 us | 20.864 us | **1.094x** |
| 32 | 23.483 us | 21.539 us | **1.090x** |
| 128 | 28.205 us | 25.583 us | **1.102x** |
| 512 | 106.161 us | 100.643 us | **1.055x** |
| 1024 | 191.119 us | 186.251 us | **1.026x** |

The 128/512/1024 rows repeat the real batch-32 captures to isolate occupancy;
they are not claimed as independently captured request distributions. Every
timed configuration first compared its selected fp32 value set with the saved
exact reference. The hot-state path passed every captured row. The cold-state
path passed `torch.topk` at batch 512, and stable request remapping now verifies
that the fused store updates `previous_topk[request_indices]` and validity.

The rewrite changes the production crossover. Direct comparison on the same
real captures found persistent-top-k/fused-GVR speedups of 1.331x, 1.274x,
1.652x, 1.060x, 1.323x, and 1.256x at batches 64, 128, 256, 384, 512, and 1024.
On a representative capture the transition was already 1.701x at batch 33;
batch 32 remains on the faster cooperative selector. Dispatch therefore now
uses GVR for rows greater than 32 and retains cooperative top-k through 32,
instead of the old conservative `>=512` guard.

Applying the isolated per-layer savings to the prior matched forced-GVR model
profiles projects baseline/new-forward ratios of approximately 0.985x, 0.989x,
1.002x, and 1.037x at batches 1, 8, 32, and 128. Thus the rewrite substantially
reduces GVR's wrapper tax but does not overturn the decision to keep cooperative
top-k at 1-32. For the corrected batch-1024 pair, the same projection changes
the measured 1.0705x speedup to about 1.072x. These are A/B projections, not a
new full-model run; the much larger operator-level gains above are the directly
measured results.

Final targeted validation after integration: `test_gvr_hints_follow_stable_request_indices`,
`test_gvr_dispatch_starts_after_cooperative_topk`, and
`test_gvr_topk_cold_start_matches_torch` all passed (3 passed, 173 deselected).
Ruff check, Ruff format, Python compilation, and `git diff --check` also passed
for the touched implementation and benchmark files.

### Measured full-model result after the structural rewrite

The projected model results above are now superseded by a real forced-all-size
measurement. A fresh baseline/GVR pair loaded the 432.9 GiB
`nvidia/GLM-5.2-NVFP4` checkpoint, used TP4, the portable DeepSeek-V3.2 model
override and V2 runner, a 100,032-token maximum context, 20 GiB of KV cache per
rank, no speculative decoding, and explicit FULL decode CUDA graphs at batches
1, 8, 32, 128, and 1024. The request contained a real 100,000-token prefix
built from repository documentation and generated 32 tokens with fixed seed 0.
The GVR eligibility guard was temporarily forced to one row for the GVR trace
and restored to the production `num_rows > 32` guard immediately afterward.

HTTP request throughput is deliberately not used: prefix admission ramps the
number of live child sequences and is not a fixed-batch decode measurement.
Instead, the table selects only NVTX ranges whose generation batch is exactly
the requested size, projects each range onto its correlated CUDA kernels, takes
the slowest of the four TP ranks for every logical step, and reports the median
critical-path GPU span. It contains 31/29/29/26/7 GVR steps and
32/29/29/26/7 baseline steps for batches 1/8/32/128/1024. The first-use and JIT
outliers therefore do not determine the result.

| Batch | Baseline forward | Forced fused GVR | GVR fixed-batch TPS | Observed speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.748 ms | 7.512 ms | 133 | **1.0314x** | 3.05% |
| 8 | 8.512 ms | 8.273 ms | 967 | **1.0289x** | 2.81% |
| 32 | 10.408 ms | 10.164 ms | 3,148 | **1.0241x** | 2.35% |
| 128 | 15.614 ms | 14.730 ms | 8,689 | **1.0600x** | 5.66% |
| 1024 | 65.711 ms | 60.302 ms | 16,981 | **1.0897x** | 8.23% |

The trace proves that this is no-fallback GVR. Every GVR rank-step has exactly
21 fused GVR launches, one for each indexer layer, and no cooperative or
filtered selector. It also has no decode hint-preparation or state-store
kernel: both operations are inside the fused launch. Every baseline rank-step
has exactly 21 selector launches: `cooperative_topk_cs16`,
`cooperative_topk_cs8`, and `cooperative_topk_cs4` at batches 1, 8, and 32,
respectively, then `FilteredTopKUnifiedKernel` at batches 128 and 1024.

The 128 and 1024 speedups are directly supported by selector attribution.
Baseline versus fused-GVR selector time is 0.954 versus 0.342 ms per rank-step
at batch 128 and 6.649 versus 1.773 ms at batch 1024. The selector therefore
saves 0.613 and 4.876 ms while the complete forward saves 0.884 and 5.409 ms.

The raw 1--32 whole-forward ratios need a stricter interpretation. Selector
times are 0.287/0.296/0.367 ms for the baseline and
0.345/0.327/0.335 ms for GVR. GVR itself is therefore 0.059 and 0.031 ms slower
at batches 1 and 8 and only 0.032 ms faster at batch 32. The remaining observed
whole-forward difference comes from non-selector kernels: the two independent
server processes selected different small-batch FP4 MoE autotune variants and
also show all-reduce/rank-skew variation. Those measured latencies are valid
real-model observations, but the 2--3% small-batch ratios are not a causal GVR
kernel speedup and do not justify moving the production guard below 33 rows.

Reports are `prof/gvr_fused_forced_all.nsys-rep` and
`prof/gvr_fused_forced_baseline_all.nsys-rep`; their exported SQLite traces are
stored beside them. The API request summaries are
`/tmp/gvr_fused_forced_api.txt` and
`/tmp/gvr_fused_forced_baseline_api.txt`.

### Autotune-disabled controlled rerun

The small-batch confound above has been removed. Both the baseline and
forced-all-size GVR servers passed `--no-enable-flashinfer-autotune`, and both
startup logs reported `KernelConfig.enable_flashinfer_autotune=False`. No
FlashInfer tuning sweep ran. All other settings were held fixed: the real
432.9 GiB GLM-5.2 NVFP4 checkpoint, TP4, portable model plus V2 runner, a real
100,000-token repository-documentation prefix, 20 GiB KV cache per rank, no
speculative decoding, and explicit FULL decode graphs at all five sizes.

As before, latency is the median fixed-batch TP critical path: select only an
exact-batch generation NVTX range, project it onto correlated GPU kernels, take
the slowest of four ranks for each logical step, then take the median. The
baseline has 33/29/29/26/7 steps and forced GVR has 32/29/29/26/7 steps at
batches 1/8/32/128/1024.

| Batch | Baseline forward | Forced GVR forward | GVR fixed-batch TPS | Speedup | Latency change |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.699 ms | 7.719 ms | 130 | **0.9974x** | 0.26% slower |
| 8 | 8.488 ms | 8.448 ms | 947 | **1.0048x** | 0.47% faster |
| 32 | 10.819 ms | 10.724 ms | 2,984 | **1.0089x** | 0.88% faster |
| 128 | 17.772 ms | 17.117 ms | 7,478 | **1.0383x** | 3.68% faster |
| 1024 | 75.307 ms | 70.056 ms | 14,617 | **1.0749x** | 6.97% faster |

Trace dispatch validation passed. Every forced-GVR rank-step contains exactly
21 fused GVR launches and no fallback selector. Every baseline rank-step
contains exactly 21 cooperative selectors at batches 1--32 or filtered
selectors at batches 128 and 1024. Baseline/GVR selector totals per rank-step
are 0.299/0.344, 0.299/0.334, 0.369/0.340, 0.956/0.347, and
6.658/1.790 ms. Thus GVR itself is 45 and 34 us slower per model forward at
batches 1 and 8, then 29 us, 609 us, and 4.868 ms faster at batches 32, 128,
and 1024. Disabling autotuning therefore confirms rather than hides the batch-1
kernel regression.

The reports are `prof/gvr_no_fi_autotune_baseline.nsys-rep` and
`prof/gvr_no_fi_autotune_forced.nsys-rep`, with SQLite exports beside them.
Request summaries are `/tmp/gvr_no_fi_autotune_baseline_api.txt` and
`/tmp/gvr_no_fi_autotune_forced_api.txt`. Future full-model GVR benchmarks
must retain `--no-enable-flashinfer-autotune` unless autotuning itself is the
subject under test.

### Can GVR be faster at every batch size?

Not as a strict property of one exact temporal kernel for every input. Exact
top-k must inspect every score, while stale or unrepresentative temporal hints
add work before an exact fallback. The one-pass cooperative baseline has no
such tax. On the real captures, a previous-hint minimum admits anywhere from
about 2,000 to almost 100,000 candidates depending on layer, which rules out a
single threshold that is uniformly cheap.

Several small-grid approaches were measured on saved real 100K logits:

- Matching the cooperative kernel's wider clusters did not transfer: GVR
  cluster 4 remained best overall; 8/16-CTA clusters paid more DSMEM reduction
  and candidate-handoff cost.
- Caching the complete per-CTA logits slice in shared memory was flat or
  slower because its footprint reduced occupancy. A two-CTA cache exceeded
  SM100's 232,448-byte shared-memory limit.
- Moving from 512 to 1024 threads improved selected batch-8 captures by
  roughly 4--8%, but remained well behind cooperative top-k.
- Removing a histogram rung or using the classic secant path saved 3--11% on
  favorable layers. Secant regressed badly on other layers, and neither path
  closed the gap.

For representative batch-1 captures, cooperative top-k averaged about
11.1 us per layer, current fused GVR 19.7 us, and the best tested GVR policy
about 18.5 us. At batch 8 the corresponding means were about 12.0, 20.9, and
19.0 us. Launch-shape and phase tuning cannot make forced GVR win there.

The practical guarantee is adaptive dispatch: retain cooperative top-k below
the measured crossover and use GVR above it. The autotune-disabled trace makes
batch 32 a validated crossover (GVR selector 0.340 versus 0.369 ms and forward
1.0089x faster), so production dispatch now starts at `num_rows >= 32` rather
than `> 32`. This makes enabling the GVR feature non-regressing across the
requested sizes: batches 1 and 8 use the baseline algorithm, while 32, 128,
and 1024 use the faster GVR path. It is deliberately a no-regression hybrid,
not a claim that forced temporal GVR is intrinsically faster at batch 1.

A genuinely faster temporal tier for batches 1--8 requires a new kernel, not
another launch-policy tweak: adapt the cooperative TMA single-pass kernel to
classify against a temporal seed while scores are resident in shared memory,
then take its existing histogram path on seed failure. That can remove GVR's
extra global scans on a hit, but it cannot guarantee strict improvement for
bad hints because detecting failure has a nonzero cost. Producer fusion with
the MQA-logits epilogue remains the only route that can create enough new
headroom for a robust strict win.

Post-change validation passed: the stable-request hint test, measured-crossover
dispatch test, and cold-start exactness test all passed (3 passed, 173
deselected). Ruff check, Ruff format, and `git diff --check` also passed.

### Worst-case slowdown

There are three different answers depending on what "worst case" means:

- With ordinary previous-token hints in the controlled real-model run, the
  worst forced-GVR result was batch 1: 7.719 versus 7.699 ms, or **0.26% e2e
  slowdown**. Production does not take that path because batches 1 and 8 use
  cooperative top-k.
- A deliberately adversarial kernel replay was also run on all six saved real
  100K-logit captures at batches 1 and 32. Each hint was a valid unique index,
  but pointed to the current row's bottom 2,048 scores--the maximally stale
  opposite of top-k. The current GVR selector, compiled with fused state store
  but direct fixed hints so every graph replay retained the bad state, remained
  exact. Its worst batch-1 latency was 27.219 versus 10.674 us for cooperative
  top-k, or **2.55x selector latency (155% slower)**. At the production
  crossover, batch 32, the worst was 33.873 versus 14.758 us, or **2.30x
  selector latency (130% slower)**.
- Those selector ratios are not whole-model ratios. Applying them to the
  selector shares from the autotune-disabled traces gives an estimated worst
  e2e penalty of about **6.0% if GVR were forced at batch 1**, and **4.4% at
  batch 32**, assuming all other kernels are unchanged. The actual hybrid has
  no batch-1 exposure, so 4.4% is the more relevant measured-adversarial
  projection for the enabled range.

The 4.4% figure is not a formal upper bound over all possible floating-point
inputs. A bad threshold bracket can activate the exact recovery path. In the
current default R0 path, the code permits one multi-threshold row scan, up to
eight bounded refinement scans, and in the coherent plateau corner up to 40
additional collapse scans plus a terminal recount before candidate collection.
That caps execution, but makes the algorithmic worst case roughly 50 row scans
before collection rather than one baseline pass. Scan count does not translate
directly to the same latency factor, and no input was found that exercised the
full budget: synthetic all-equal and wide-boundary-tie rows actually terminated
quickly and beat the baseline. Therefore the defensible bounds today are
**2.30x selector / about 4.4% projected e2e as the worst measured adversarial
case in the production-enabled range**, while the absolute adversarial latency
maximum remains uncharacterized.

### Beyond top-k: reducing the indexer producer

The autotune-disabled forced-GVR trace changes the optimization target at large
batch. On one representative rank-step, the 21 indexer layers spend:

| Batch | Paged-MQA logits | GVR top-k | Combined | Forward |
|---:|---:|---:|---:|---:|
| 1 | 0.156 ms | 0.360 ms | 0.516 ms | 7.719 ms |
| 8 | 0.307 ms | 0.340 ms | 0.647 ms | 8.448 ms |
| 32 | 0.742 ms | 0.350 ms | 1.092 ms | 10.724 ms |
| 128 | 2.540 ms | 0.351 ms | 2.891 ms | 17.117 ms |
| 1024 | 19.059 ms | 1.792 ms | 20.851 ms | 70.056 ms |

At batch 1024 the standalone selector is only 2.6% of the forward, while the
complete score producer plus selector is 29.8%. Deleting top-k alone therefore
cannot yield a drastic additional win. The next design must avoid dense score
production, or at least fuse selection into the producer epilogue so the
`[1024, 100032]` fp32 logits tensor is not materialized and reread.

#### Numeric-cutoff epilogue

The previous exact kth score is a poor single speculative threshold, but a
small additive ladder is substantially more robust. For each of 123
consecutive real rows, the exact admissible threshold interval was computed as
the thresholds producing a count in `[2048, 6144]`. Five offsets from the
previous kth score stab all 123 intervals:

```text
-2.996635437, -1.795748830, -1.348399222, -0.169873834, +1.206140101
```

Three offsets cover 95.1% of the rows. This is evidence for a speculative
producer epilogue, not a production calibration: the offsets are fitted to
only three captured layers and two adjacent decode steps.

A naive bounded candidate append is not sufficient. The lowest rung admits a
median 4,133 scores but a p95 of 93,613 and maximum of 99,030. A fused epilogue
would therefore need per-rung bins, a count-only first stage followed by sparse
rescoring, or an exact dense fallback; reserving one buffer for the lowest rung
nearly recreates the full logits tensor. Rounded decimal offsets also reduced
coverage to 95.9%, demonstrating that equality/tie handling must use exact
float endpoints or conservative `nextafter` adjustment.

#### Temporal and page pruning

Rescoring only the previous top-k is too approximate for this workload. Exact
top-k overlap across the 123 rows averages 1,722/2,048 (84.1%), with a minimum
of 1,030. Expanding the previous winners to their 64-token KV pages improves
mean recall to 97.2% at about 8,107 candidate tokens, but worst-row recall is
only 72.0%. Including adjacent pages raises mean recall to 98.5% at 13,519
candidates and still has a 75.3% minimum. These are potentially useful
quality/performance modes, but they are not logically exact.

Keeping previous per-page score maxima is stronger: selecting the best 256
old pages (at most 16,384 tokens) gives 99.37% mean current-top-k recall and
92.48% minimum; 512 pages (32,768 tokens) gives 99.95% mean and 98.19% minimum.
One captured layer needs 768 pages, or roughly half the context, for 100%
observed recall. A global exact temporal score-drift bound is not useful:
even an oracle bound based on the actual largest score increase retains a
median 19,648 tokens and all 100,032 tokens at p95; for the unstable layer it
retains about 93K tokens on average.

#### Exact vector-space bounds

Real quantized producer inputs were captured for two finite indexer calls. The
physical FP8 cache layout was reconstructed exactly: each 64-token page holds
`64 * 128` FP8 key bytes followed by 64 fp32 scales. Recomputing
`sum_h w_h * relu(q_h dot k)` from the captured Q, K, scales, and weights
matched DeepGEMM logits within `2.86e-6`.

This allowed the proposed vector-space pruning test to use real keys rather
than final-score proxies. Conservative axis-aligned boxes around mini-k-means
clusters were not selective enough. On the representative call, 256, 1,024,
and 4,096 boxes retained essentially 100% of keys; 8,192 retained 99.1%,
16,384 retained 95.1%, and 32,768 still retained 76.4%. Computing 32-head query
bounds for 32,768 boxes already costs about 32.8% of the original dense QK
work before scoring survivors. Sphere bounds were looser. Chronological groups
of up to eight keys also retained 100%. Exact value-space pruning is therefore
rejected for this workload.

#### Reduced-precision logits with exact repair

FP16 rounding is monotonic, so an approximate FP16 top-k plus exact repair of
the boundary tie bucket can be logically exact. Across 246 saved real rows,
raw FP16 top-k differed from FP32 on 75.2% of rows but missed only 1.67 indices
on average; the FP16 cutoff bucket had median/p95/max sizes 10/20/25. BF16 was
less attractive: 12.92 misses on average and tie buckets 79.5/136/150. GVR's
dense scan itself was 1.41x faster on FP16 at batch 1024.

The producer negates that benefit today. DeepGEMM accepts fp32 and BF16 output,
not FP16, and a direct real-Q/K measurement gave fp32/BF16 paged-MQA latencies
of 22.397/22.610 us at batch 1, 34.944/34.809 at 32, 136.467/132.431 at 128,
and 1,040.869/1,168.626 at 1024. BF16 is 11% slower at the important large
batch. Adding FP16 output support plus exact boundary repair remains possible,
but reduced output bandwidth alone is not a demonstrated producer win.

#### Shared-prefix producer scheduling

The strongest new direction is exact KV reuse across query rows. DeepGEMM's
SM100 varlen scheduler treats a run of equal indices as one logical request and
processes up to four query rows against each KV tile. For requests sharing a
long prefix, run the common complete pages with grouped indices, run each
request's private final page with unique indices, and copy the at-most-64 tail
scores into the grouped output. The current GVR selector then consumes the
ordinary dense layout unchanged.

`benchmarks/kernels/benchmark_gvr_shared_prefix.py` exercises this with the
captured real FP8 Q/K/scale/weight tensors. `--private-tail` deliberately maps
the final page of each row to a different physical page, so equality is not an
artifact of repeating one block table. Both the two-DeepGEMM segmented result
and the stitched dense result are bit-identical to an ordinary unique-index
full producer. The final command was:

```bash
.venv/bin/python benchmarks/kernels/benchmark_gvr_shared_prefix.py \
  /tmp/gvr_indexer_inputs/indexer_call21.pt \
  --batches 1,8,32,128,1024 --repeats 200 --private-tail
```

One representative stitched run measured 18.655/62.581 us at batch 1,
16.303/57.571 at 8, 34.636/56.652 at 32, 133.637/78.834 at 128, and
1,195.103/550.892 at 1024 (ordinary/stitched). Three additional 300-repeat
runs made the useful tiers stable: **1.70--1.74x at batch 128** and
**2.21--2.25x at batch 1024**. The extra launches lose below batch 128, so this
must be an adaptive tier rather than an all-batch replacement.

`benchmark_gvr_shared_prefix_sweep.py` then varied the shared fraction while
rotating every suffix block table row to different physical pages. All cells
remained bit-exact. The crossover depends strongly on batch size and on
DeepGEMM's shape heuristic: batch 128 was 0.904x at 60% sharing, approximately
parity at 65%, and 1.17x at 70%; batch 256 was 0.88x/1.16x at 50%/70%; batch
512 was 1.07x/1.33x at 50%/70%; batch 1024 was 0.95x at 25%, roughly parity
around 35--40%, and 1.10x at 50%. At 90% sharing the 128/256/512/1024 gains
were 1.28x/1.36x/1.61x/1.71x. Because several intermediate lengths select
non-monotonic DeepGEMM configurations, conservative initial dispatch should
require at least 85% sharing at batch 128, 70% at 256, and 50% at 512/1024,
then replace those hand thresholds with offline-tuned shape decisions.

Applying the conservative ends of those ranges to the measured 21-layer
producer totals projects the batch-128 forward from 17.117 to about 16.08 ms
(**1.06x additional speedup**) and the batch-1024 forward from 70.056 to about
59.8 ms (**1.17x additional speedup**). Relative to the original 75.307 ms
batch-1024 baseline, GVR plus shared-prefix production would be about
**1.26x**. These are projections, not yet full-model measurements.

A one-launch producer scheduler was also prototyped. It processed the grouped
prefix and then unpaired suffix rows in one linear task stream and remained
bit-exact, but the suffix was assigned to the last few SMs. With 1,024 rows it
regressed to 31 ms despite lower nominal work. A correct one-launch rewrite
must interleave prefix and suffix work per query block or give every SM a
separate range for both phases; simply appending suffix tasks is invalid for
load balance. That prototype was removed.

Production integration still needs a CUDA-graph-compatible common-prefix
metadata path. vLLM already computes the number of common cache blocks, but
ordinary cascade attention disables FULL CUDA graphs, and the indexer builder
currently opts out of cascade metadata. The safe implementation is a separate
indexer prefix tier captured at batch 128 and above: persistent prefix/tail
lengths and schedule buffers, grouped/unique DeepGEMM calls, the 64-score
scatter, and fallback to the existing dense producer when the common prefix is
too short. Until that graph variant exists and is measured end to end, the
shared-prefix result remains a validated kernel prototype rather than enabled
production behavior.

### Native FP16 DeepGEMM paged-MQA output (in progress)

The pinned DeepGEMM implementation was extended to support FP16 output from
`fp8_fp4_paged_mqa_logits`. Its paged-MQA kernels were already templated on the
output type and accumulate scores in FP32; two host-side omissions prevented
FP16 use: the scalar-type-to-CUTLASS mapper had no `torch::kFloat16` case, and
the public paged API admitted only FP32/BF16. The maintained vLLM patch adds
`cutlass::half_t` dispatch and permits FP16. CMake applies the patch to both
fetched and `DEEPGEMM_SRC_DIR` source trees, while accepting an already-patched
tree.

The vLLM compatibility wrapper now exposes `logits_dtype`, defaulting to FP32.
`VLLM_GVR_FP16_LOGITS=1` is an explicit approximate-accuracy mode: for an
eligible NVIDIA indexer decode, DeepGEMM writes FP16 logits and the existing
FP16-capable GVR selector is used at every batch size. The opt-in is separate
from `VLLM_USE_GVR_TOPK` because changing score precision can change selected
indices; FP32 behavior remains the default.

The rebuilt extension was tested on an NVIDIA GB200 with Torch 2.13.0+cu130:

```text
.venv/bin/python -m pytest \
  tests/kernels/attention/test_deepgemm_attention.py::\
test_deepgemm_fp8_fp4_paged_mqa_logits -v

2 passed in 13.12s
```

The test runs the real FP8 paged-cache kernel for `next_n=1` and `next_n=2`
with both FP32 and FP16 logits, checks the requested output dtype, and compares
each result with the dequantized PyTorch reference. The FP16 GVR cold-start
correctness test also passed against `torch.topk` (2 parametrized cases total
for FP32/FP16).

#### Real-model accuracy

The full 1,319-example GSM8K 5-shot evaluation used the real 432.9 GiB
`nvidia/GLM-5.2-NVFP4` checkpoint, TP4, temperature zero, 128 concurrent
requests, and FP16 logits plus GVR. It scored **0.9477 strict** (1,250/1,319,
stderr 0.0061) and **0.9484 flexible** (1,251/1,319, stderr 0.0061). The prior
FP32 baseline scored 0.9348/0.9356 and exact FP32 GVR scored 0.9363/0.9363.
FP16 is numerically higher in this run; the strict difference from the FP32
baseline is about 1.4 combined standard errors, so the defensible conclusion
is no observed task-accuracy regression, not a claimed accuracy improvement.
The result is in
`gsm8k_gvr_fp16/GLM-5.2/results_2026-08-13T22-33-19.070192.json`.

#### Real-model fixed-batch performance

The performance run retained every controlled condition from the earlier
autotune-disabled comparison: TP4, portable DeepSeek-V3.2 model override and V2
runner, real 100,000-token repository-documentation prefix, 20 GiB KV cache per
rank, no speculative decoding, and FULL decode graphs captured at batches
1/8/32/128/1024. `--no-enable-flashinfer-autotune` was explicit and the engine
reported `enable_flashinfer_autotune=False`. GVR was enabled at every size.

Latency is the median fixed-batch graph duration on the TP critical path. Each
NVTX range is matched to its `cudaGraphLaunch` correlation ID and corresponding
Nsight `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` interval; the slowest of the four ranks
is taken for each logical step. There are 33/29/29/26/7 steps at batches
1/8/32/128/1024.

| Batch | Original baseline | FP32 GVR | FP16 GVR | FP16 fixed-batch TPS | FP16 vs baseline | FP16 vs FP32 GVR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.699 ms | 7.719 ms | **7.420 ms** | 135 | **1.0376x** | **1.0403x** |
| 8 | 8.488 ms | 8.448 ms | **8.033 ms** | 996 | **1.0566x** | **1.0517x** |
| 32 | 10.819 ms | 10.724 ms | **10.339 ms** | 3,095 | **1.0464x** | **1.0372x** |
| 128 | 17.772 ms | 17.117 ms | **16.705 ms** | 7,662 | **1.0639x** | **1.0247x** |
| 1024 | 75.307 ms | 70.056 ms | **68.963 ms** | 14,849 | **1.0920x** | **1.0158x** |

Unlike exact FP32 GVR, the FP16 combination is faster than the original
selector baseline at every measured size. It reduces complete forward latency
by 3.62%, 5.36%, 4.44%, 6.00%, and 8.42%, respectively. At batch 1024, native
FP16 adds 1.56% over FP32 GVR and raises the total baseline speedup from 1.075x
to **1.092x**. At small batches the avoided 400 KiB-per-row FP32 materialization
is enough to overcome GVR's former one-row/tiny-grid disadvantage.

Dispatch validation is explicit in the trace. The paged producer demangles as
`deep_gemm::sm100_paged_mqa_logits<..., cutlass::half_t, float, ...>` and the
selector symbol contains `tensorptrf16`. The report and SQLite export are
`prof/gvr_fp16_no_fi_autotune.nsys-rep` and
`prof/gvr_fp16_no_fi_autotune.sqlite`; the request-ramp summary is
`/tmp/gvr_fp16_no_fi_autotune_api.txt`. As in earlier runs, its 709.5 tok/s at
batch 1024 is admission-ramp throughput and must not replace the fixed-batch
14,849 tok/s derived from graph intervals.

There is not yet a native "FP16 without GVR" result. vLLM's cooperative and
persistent sparse-indexer selectors both reject non-FP32 logits, store the
input as `const float*`, and dispatch `FilteredTopKRaggedTransform<float, ...>`.
The filtered implementation is nominally dtype-templated, but currently only
defines `FilteredTopKTraits<float>`; the cooperative and persistent radix paths
are more deeply FP32-specific. Therefore disabling GVR also disables FP16
production in the current integration. An FP16-producer/FP32-selector ablation
would require materializing a full FP32 conversion before top-k, while the
meaningful native comparison requires adding and validating FP16 baseline
selector kernels. The measured 1.56% batch-1024 improvement from FP32 GVR to
FP16 GVR combines producer/output-traffic and selector effects and cannot be
attributed to FP16 DeepGEMM alone.

#### Isolated FP16 producer effect

A producer-only benchmark separates DeepGEMM from GVR. It uses the captured
real indexer Q/K/scale/weight tensors from `indexer_call20.pt` at sequence
length 100,001, repeats that real row to each tested batch, constructs the
normal paged-MQA schedule, and alternates FP32 and FP16 output measurements for
seven rounds. CUDA-event medians are:

| Batch | FP32 output | FP16 output | FP16 producer speedup |
|---:|---:|---:|---:|
| 1 | 20.357 us | 20.368 us | 0.9995x |
| 8 | 19.739 us | 19.507 us | 1.0119x |
| 32 | 31.002 us | 31.350 us | 0.9889x |
| 128 | 112.982 us | 114.321 us | 0.9883x |
| 1024 | 884.408 us | 887.659 us | 0.9963x |

FP16 therefore does not materially accelerate the paged-MQA producer. The
kernel performs the same low-precision Q/K products, FP32 accumulation, ReLU,
and weighted head reduction; only its final output conversion and stores are
narrower. For batch 1024 and width 100,032, FP16 saves about 205 MB of output
stores, but the operation logically reads roughly 13.5 GB of 132-byte KV
records across the rows and performs the same dense dot-product work. Output
bandwidth is not the limiting resource, and the extra FP32-to-FP16 conversion
can offset the smaller stores.

Consequently, the 1.56% full-forward improvement between FP32 GVR and FP16 GVR
must not be described as a DeepGEMM producer gain. It primarily reflects the
cheaper FP16 GVR scan plus normal whole-graph variation; the isolated producer
contributes approximately zero improvement.

#### Raw GVR top-k FP16 microbenchmark

The selector was also isolated from both DeepGEMM and the external hint/state
kernels. Six real score, hint, and sequence-length captures were used at each
native batch 1, 8, and 32; the six 32-row captures were repeated to construct
larger batches. Every row has width 100,032 and requests top-2,048. Each
selector was captured as ten consecutive CUDA-graph nodes, warmed with three
graph replays, and timed for ten further replays with CUDA events. The FP32
baseline uses the production dispatch: `cooperative_topk` through batch 32 and
`persistent_topk` above it. Workspaces are allocated outside the graph and the
timed region.

Both GVR dtypes were checked by selected-value multiset against `torch.topk`,
with columns beyond each sequence length masked. FP16 exactness here means
exact top-k of the rounded FP16 scores, not exact agreement with the original
FP32 ordering.

| Batch | FP32 baseline | FP32 GVR | Baseline / GVR | FP16 GVR | Baseline / GVR |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.399 us | 17.147 us | 0.665x | 16.635 us | 0.685x |
| 8 | 12.188 us | 19.621 us | 0.621x | 18.628 us | 0.654x |
| 32 | 15.287 us | 19.925 us | 0.767x | 18.974 us | 0.806x |
| 128 | 32.627 us | 24.200 us | 1.348x | 22.484 us | 1.451x |
| 256 | 64.549 us | 37.161 us | 1.737x | 35.794 us | 1.803x |
| 512 | 133.069 us | 91.498 us | 1.454x | 60.840 us | 2.187x |
| 1024 | 234.171 us | 163.262 us | 1.434x | 103.627 us | **2.260x** |
| 2048 | 460.694 us | 274.208 us | 1.680x | 192.202 us | **2.397x** |
| 4096 | 888.973 us | 486.448 us | 1.827x | 363.714 us | **2.444x** |

The production baseline wins at batches 1--32. Both GVR dtypes cross over by
batch 128. At batch 1024, FP16 GVR is 2.26x faster than non-GVR FP32 and 1.58x
faster than FP32 GVR. FP16 halves dense score traffic, but GVR also contains
dtype-independent threshold, histogram, candidate, and exact-ranking phases.
Narrower score loads and the FP16 occupancy tier matter most from batch 512;
at still larger multi-wave grids the FP16-over-FP32-GVR ratio settles to
1.34--1.43x even while both increasingly outperform persistent top-k.

The small-batch loss is a launch-shape and fixed-work effect, not missing CUDA
graphs. For width 100,032 on the 132-SM GB200, the production cooperative
selector uses 1,024-thread CTAs and cluster sizes 16, 8, and 4 at batches 1, 8,
and 32. GVR's current policy uses cluster size 4 and 512-thread CTAs at all
three sizes:

| Batch | Baseline CTAs | GVR CTAs | Baseline threads/CTA | GVR threads/CTA |
|---:|---:|---:|---:|---:|
| 1 | 16 | 4 | 1,024 | 512 |
| 8 | 64 | 32 | 1,024 | 512 |
| 32 | 128 | 128 | 1,024 | 512 |

The cooperative baseline is deliberately a few-rows algorithm: it splits one
row over many CTAs to occupy the GPU. This is why its latency increases only
from 11.4 us at one row to 15.3 us at 32 rows. GVR first reads and reduces the
2,048 temporal hints, constructs histogram rungs, scans the score row for
threshold admission, collects candidates, and performs exact rank/scatter.
Those fixed phases and cluster synchronization cost several microseconds even
when the score scan is small enough to remain cache/latency dominated. FP16
therefore saves only 0.5--1.0 us at batches 1--32: halving 0.4--12.8 MB of score
input is not yet the dominant term.

At batch 128 the baseline must switch to the one-CTA-per-row persistent/filtered
kernel. GVR also switches to one CTA per row, now has enough independent rows
to fill the GPU, and its temporal threshold avoids enough baseline radix/filter
work to win. At batch 1024 the dense score tensor is approximately 410 MB in
FP32 versus 205 MB in FP16, so score traffic is finally large enough for FP16
GVR's bandwidth and occupancy advantages to dominate its fixed phases.

CUDA graphs remove CPU launch and graph-construction overhead; they do not
remove GPU kernel launch latency, limited CTA parallelism, cluster barriers, or
the algorithm's internal phases. All numbers in the table already include the
same CUDA-graph treatment.

Matching the baseline launch shape was tested directly rather than assumed to
help. GVR was forced to the baseline's cluster/thread configurations: CS16 x
1,024 threads at batch 1, CS8 x 1,024 at batch 8, and CS4 x 1,024 at batch 32.
The same six real captures and ten-node CUDA graphs were used:

| Batch | FP32 baseline | Default FP32 GVR | Matched FP32 GVR | Default FP16 GVR | Matched FP16 GVR |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.431 us | 17.185 us | 17.920 us | 16.683 us | 17.867 us |
| 8 | 12.195 us | 19.605 us | **18.562 us** | 18.585 us | **18.105 us** |
| 32 | 15.305 us | 19.996 us | 20.053 us | 18.996 us | 19.189 us |

At batch 1, matching regresses FP32/FP16 GVR by 4.3%/7.1%. At batch 8 it
reduces their latency by 5.3%/2.6%, and at batch 32 it is neutral to 1% slower.
Even the best matched result remains much slower than cooperative top-k.

More CTAs are not free for this GVR implementation. Each cluster member owns a
score slice, but counts, candidate chunks, and exact completion must be
aggregated through cluster barriers and distributed shared-memory peer reads;
some final work remains leader-heavy. Increasing CS from 4 to 16 reduces each
batch-1 CTA's slice to only about 6,252 scores while multiplying peer
coordination. Increasing from 512 to 1,024 threads also gives too little work
per thread to recover that cost. The cooperative baseline was architected
around CS16/8/4, so copying its launch dimensions does not copy its efficiency.

Production GVR dispatch is therefore unchanged. A genuinely competitive
small-batch GVR would need a different cooperative algorithm--for example,
temporal threshold seeding inside the baseline's existing cooperative
histogram path--rather than a launch-parameter substitution.

The FP32 persistent baseline matched the reference on every tested row. The
cooperative baseline mismatched the exact selected-value set on 7 of the 192
captured batch-32 rows, consistent with the previously documented issue; its
timing is still reported because it is the deployed non-GVR selector.

This is a steady-state kernel microbenchmark, not another model-forward result.
Repeating captured rows preserves six real score distributions but does not
reproduce the complete distribution of an actual 1,024-request batch. The
measured 68.963 ms FP16 model forward remains the authoritative e2e result.

## Native-FP16 production baseline experiment

The next comparison removes a remaining asymmetry: the production
`cooperative_topk`/`persistent_topk` baseline previously accepted only FP32,
while the fastest GVR tier reads scores natively as FP16. The baseline is being
generalized to read FP16 directly, without an FP16-to-FP32 staging kernel or a
full temporary score tensor.

The cooperative kernel now templates its input, TMA transfers, and resident
score buffers on FP32/FP16; comparisons and tie metadata remain FP32 after an
exact half-to-float conversion. The filtered persistent kernel uses the
sign-monotonic 16-bit FP16 representation directly: its coarse pass consumes
the high byte and one refinement consumes the low byte, so it selects the
exact top-k of the rounded FP16 input. The existing FP32 paths and dispatch
thresholds remain unchanged.

The modified `topk.cu` and `cooperative_topk.cu` compiled successfully for
SM100. The required full editable-install build later failed in the unrelated
optional QuTLASS target because it included non-stable Torch headers while
`TORCH_TARGET_VERSION` was defined. Building and installing only vLLM's
`_C_stable_libtorch` target succeeded. The focused padded-stride pytest suite
then passed all 8 FP32/FP16 x cooperative/persistent x K=512/2048 cases.

The real-data microbenchmark used the same six captures at each native batch
and the same CUDA-graph protocol as the preceding table: width 100,032,
K=2,048, ten operations per graph, three warm replays, and ten timed replays.
Batches above 32 repeat the six real batch-32 captures. FP16 baseline and FP16
GVR were both checked against exact `torch.topk` of the rounded FP16 scores.

| Batch | FP32 baseline | Native FP16 baseline | FP32 / FP16 | FP16 GVR | FP16 baseline / GVR |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.478 us | 11.269 us | 1.019x | 16.671 us | 0.676x |
| 8 | 12.224 us | 11.983 us | 1.020x | 18.694 us | 0.641x |
| 32 | 15.163 us | 14.737 us | 1.029x | 19.007 us | 0.775x |
| 128 | 32.882 us | 29.234 us | 1.125x | 23.804 us | 1.228x |
| 256 | 65.797 us | 55.409 us | 1.187x | 35.601 us | 1.556x |
| 512 | 133.336 us | 106.838 us | **1.248x** | 60.842 us | 1.756x |
| 1024 | 234.816 us | 189.691 us | **1.238x** | 103.607 us | **1.831x** |
| 2048 | 460.481 us | 372.695 us | **1.236x** | 192.498 us | **1.936x** |
| 4096 | 891.251 us | 725.197 us | **1.229x** | 363.372 us | **1.996x** |

Native FP16 helps the baseline substantially once the one-CTA-per-row
persistent kernel is bandwidth relevant, but it does not close the algorithmic
gap. Its large-batch gain is 1.23-1.25x rather than 2x because the baseline
still performs coarse histogramming, candidate filtering, and exact radix
refinement; only score traffic is halved. At batch 1-32, the cooperative path
is dominated by launch and synchronization, so FP16 saves only 1.9-2.9%.

The result strengthens the hybrid policy: native FP16 cooperative top-k is the
fastest measured exact selector for batches 1-32, while FP16 GVR crosses over
at batch 128 and grows from 1.23x to 2.00x faster than the native-FP16 baseline.
The cooperative baseline retained its previously documented correctness issue
on 8/192 real batch-32 rows in both dtypes. Persistent FP16 and GVR had zero
selected-value mismatches on every tested row.

A longer 50-operation-graph x 50-replay confirmation at the requested batches
measured FP32/native-FP16 baseline latencies of 11.173/10.995 us (B1),
11.891/11.643 us (B8), 14.800/14.414 us (B32), 32.729/28.840 us (B128), and
234.576/189.486 us (B1024). These correspond to native-FP16 speedups of
1.016x, 1.021x, 1.027x, 1.135x, and 1.238x and agree with the original sweep.

## 2026-08-14: requested real-data batch/KV matrix

Q1 was clarified to mean the aggregate percentage of a complete decode
forward occupied by top-k. GLM-5.2-NVFP4 invokes the sparse-indexer top-k 21
times per forward, so the measured share is `21 * top-k kernel latency /
full-forward latency`. With FP32 baseline top-k and matched real-model forward
measurements, the shares are:

| KV length | B1 | B8 | B32 | B128 | B1024 |
|---:|---:|---:|---:|---:|---:|
| 10K | 1.50% | 1.43% | 1.17% | 0.83% | 1.38% |
| 50K | 3.04% | 2.83% | 2.76% | 3.31% | 5.74% |
| 100K | 3.13% | 3.04% | 2.94% | 3.67% | 6.16% |
| 200K | 3.68% | 3.87% | 4.55% | 7.06% | 10.34% |

This is top-k kernel time only, excluding indexer-logit GEMMs. Even a
zero-latency top-k would therefore yield only 1.01-1.12x decode-forward
speedup in this matrix; the largest theoretical ceiling is 1.115x at
200K/B1024. The matched forced-GVR end-to-end run kept FlashInfer autotuning
disabled.

The matched run completed with full-decode CUDA graphs, real model weights,
real document prompts, and 54-242 exact decode samples per cell. Median
baseline/GVR forward latencies and speedups were:

| KV length | B1 | B8 | B32 | B128 | B1024 |
|---:|---:|---:|---:|---:|---:|
| 10K | 7.065/7.189 ms (0.983x) | 7.658/7.802 ms (0.982x) | 9.518/9.649 ms (0.986x) | 14.289/14.336 ms (0.997x) | 52.289/51.843 ms (1.009x) |
| 50K | 7.186/7.219 ms (0.995x) | 7.828/8.180 ms (0.957x) | 9.980/10.248 ms (0.974x) | 15.741/15.376 ms (1.024x) | 62.879/59.501 ms (1.057x) |
| 100K | 7.220/7.261 ms (0.994x) | 7.958/7.976 ms (0.998x) | 10.385/10.243 ms (1.014x) | 17.235/16.628 ms (1.036x) | 74.715/69.536 ms (1.074x) |
| 200K | 7.307/7.323 ms (0.998x) | 8.183/8.158 ms (1.003x) | 11.191/10.924 ms (1.024x) | 20.513/19.085 ms (1.075x) | 99.351/89.787 ms (1.107x) |

GVR is not useful at B1 and generally regresses B8/B32 at short contexts. Its
meaningful gains start around B128 for 50K+ contexts and reach 10.65% at
200K/B1024. The temporary capture, force-all-rows, and forward-timing hooks
were removed after measurement. During cleanup, the non-GVR selector branch
was fixed to refresh GVR state so later batch-size crossover into GVR does not
consume stale state; the GVR kernel itself already fuses this update.

### Reconciliation with the earlier GVR results

The new result does not show an apples-to-apples 100K e2e regression. With
FlashInfer autotuning disabled, old/new speedups at B1/B8/B32/B128/B1024 are
0.9974/0.9943x, 1.0048/0.9978x, 1.0089/1.0139x, 1.0383/1.0365x, and
1.0749/1.0745x. The differences are at most 0.7 percentage points, and the
important B128/B1024 results reproduce almost exactly.

The broader matrix looks worse for three reasons. First, it deliberately
forces GVR into short-context and small-batch regimes where the specialized
cooperative baseline needs only 5-15 us per layer and GVR's fixed
guess/histogram/refinement machinery cannot amortize its work. Earlier
production dispatch avoided those cells. Second, earlier apparent 2-3%
small-batch whole-forward gains came from different FP4 MoE autotune choices
and all-reduce/rank-skew variation; the autotune-disabled trace had already
reduced those claims to approximately neutral. Third, GVR is data-dependent.
The new kernel B128/B1024 rows duplicate first-layer native-B32 captures,
whereas the old trace and the new e2e test execute independent real rows across
all 21 layers. Repetition preserves selector occupancy but not the large-batch
distribution of hint quality and fallback behavior. This explains why the
200K repeated-row kernel result can be poor while the authoritative real
B1024 model forward improves by 10.65%.

Absolute old/new forward latencies also use different aggregation: the older
Nsight analysis took each step's slowest TP rank, while the new lightweight
timer records rank 0 around CUDA-graph replay. Their matching 100K ratios show
that this accounting difference is not hiding a GVR regression.

### Kernel-level reconciliation and 200K diagnosis

At the kernel level, GVR itself did not become slower at the previously tested
100K point. The earlier fused-GVR latencies at B1/B8/B32/B128/B1024 were
19.575/20.864/21.539/25.583/186.251 us; the new raw-selector measurements are
14.461/20.834/20.372/24.964/139.254 us. The poor-looking small-batch ratios
come from comparing against the much cheaper cooperative baseline, not from a
GVR latency regression. The earlier large speedups were specifically for 100K
B64+ against persistent top-k; they did not claim GVR beat cooperative top-k
at B1-B32 or cover 10K/50K/200K.

The anomalous default 200K/B1024 result was isolated to the R0 threshold
policy. Replaying the same two real captures gave:

| GVR policy | Mean latency | Versus 489.199-us baseline |
|---|---:|---:|
| Default rungs `(q=.60, .35)` + virtual seed | 584.765 us | 0.837x |
| Default + 1024 threads | 507.010 us | 0.965x |
| Secant only | 416.885 us | 1.173x |
| Three rungs `(q=.35, .05, .01)` | **281.509 us** | **1.738x** |

The default grows superlinearly from 139.254 us at 100K to 584.765 us at
200K because its rungs produce an unfavorable admission/fallback path on
these score distributions. Thread-count tuning recovers only part of the loss;
changing the algorithmic thresholds recovers most of it. This is the inverse
of the earlier six-capture 100K result, where the three-rung variant was slower
than the default. A single threshold set therefore does not generalize across
KV lengths, which is exactly the data sensitivity the expanded matrix was
intended to reveal. A length/data-aware policy or cheap runtime admission
diagnostic is needed before treating the original default-GVR table as the
best achievable kernel result.

### Runtime-adaptive rung fix and B64 kernel result

A union ladder was tested first. It handled both distributions but required
four or five threshold columns and left 4-9% on the table. The implemented
fix retains three columns and selects their meaning from the device-side
runtime sequence length, so it works inside one captured CUDA graph:

- below 32K and from 64K (inclusive) to 128K (exclusive):
  `(q=.60, .35, pmean)`;
- from 32K (inclusive) to 64K (exclusive), and at 128K or longer:
  `(q=.35, .05, .01)`.

Both branches use the same exact fallback and exact Phase 4. Every timed row
was checked against the selected FP32 value set from `torch.topk`. The final
50-operation/100-replay protocol is used through B64; B128/B1024 use
20 operations and 50 replays. B64 and larger repeat native real B32 captures,
so they measure kernel scaling rather than independently captured request
distributions.

| KV length | Batch | FP32 baseline | Fixed GVR | Speedup (baseline/GVR) |
|---:|---:|---:|---:|---:|
| 10K | 1 | 5.038 us | 9.727 us | 0.518x |
| 10K | 8 | 5.187 us | 9.908 us | 0.524x |
| 10K | 32 | 5.324 us | 9.804 us | 0.543x |
| 10K | 64 | 5.509 us | 10.167 us | 0.542x |
| 10K | 128 | 5.637 us | 9.935 us | 0.567x |
| 10K | 1024 | 34.187 us | 42.717 us | 0.800x |
| 50K | 1 | 10.400 us | 16.389 us | 0.635x |
| 50K | 8 | 10.506 us | 18.026 us | 0.583x |
| 50K | 32 | 13.055 us | 17.802 us | 0.733x |
| 50K | 64 | 24.424 us | 18.609 us | **1.312x** |
| 50K | 128 | 24.747 us | 18.309 us | **1.352x** |
| 50K | 1024 | 171.590 us | 98.077 us | **1.750x** |
| 100K | 1 | 10.975 us | 14.461 us | 0.759x |
| 100K | 8 | 11.538 us | 20.903 us | 0.552x |
| 100K | 32 | 14.435 us | 20.493 us | 0.704x |
| 100K | 64 | 29.761 us | 22.898 us | **1.300x** |
| 100K | 128 | 30.079 us | 25.136 us | **1.197x** |
| 100K | 1024 | 219.171 us | 140.239 us | **1.563x** |
| 200K | 1 | 12.834 us | 18.305 us | 0.701x |
| 200K | 8 | 15.128 us | 19.772 us | 0.765x |
| 200K | 32 | 24.167 us | 20.311 us | **1.190x** |
| 200K | 64 | 54.677 us | 26.004 us | **2.103x** |
| 200K | 128 | 67.016 us | 33.519 us | **1.999x** |
| 200K | 1024 | 488.980 us | 283.554 us | **1.724x** |

The speedup direction matters: `0.518x` means GVR is `1 / 0.518 = 1.93x` as
slow in latency, not that its latency is 51.8% of baseline. For B1/B8/B32,
fixed-GVR latency is 1.93/1.91/1.84x baseline at 10K,
1.58/1.72/1.36x at 50K, 1.32/1.81/1.42x at 100K, and
1.43/1.31/0.84x at 200K. The 200K/B32 cell is the only win in that range.

B64 is a clear crossover at every measured context of 50K or longer. The fix
changes 50K/B1024 from 1.400x to 1.750x and 200K/B1024 from 0.836x to 1.724x,
while preserving the 100K behavior. It does not solve short-row fixed cost:
10K loses at every batch, and B1-B32 still lose at 50K/100K.

### Fixed-GVR separate-server e2e rerun

The two server processes used matching real GLM-5.2-NVFP4 weights and document
inputs, TP4, full-decode CUDA graphs captured at
B1/B8/B32/B64/B128/B1024, and disabled FlashInfer autotuning. GVR was forced
at all sizes for measurement. Each cell is the median of 56-242 rank-0
full-forward CUDA-event samples; the 200K/B1024 row has 127 samples. Baseline
B64 was rerun with the new exact B64 graph. Other baseline cells reuse the same
configuration's earlier, separate server run.

| KV length | Batch | Baseline | Fixed GVR | Speedup | Change |
|---:|---:|---:|---:|---:|---:|
| 10K | 1 | 7.065 ms | 7.537 ms | 0.937x | -6.26% |
| 10K | 8 | 7.658 ms | 8.125 ms | 0.943x | -5.75% |
| 10K | 32 | 9.518 ms | 9.663 ms | 0.985x | -1.50% |
| 10K | 64 | 11.543 ms | 11.666 ms | 0.989x | -1.06% |
| 10K | 128 | 14.289 ms | 14.334 ms | 0.997x | -0.32% |
| 10K | 1024 | 52.289 ms | 51.714 ms | 1.011x | +1.11% |
| 50K | 1 | 7.186 ms | 7.222 ms | 0.995x | -0.50% |
| 50K | 8 | 7.828 ms | 7.874 ms | 0.994x | -0.58% |
| 50K | 32 | 9.980 ms | 9.932 ms | 1.005x | +0.48% |
| 50K | 64 | 12.535 ms | 12.203 ms | 1.027x | +2.72% |
| 50K | 128 | 15.741 ms | 15.781 ms | 0.997x | -0.25% |
| 50K | 1024 | 62.879 ms | 59.484 ms | 1.057x | +5.71% |
| 100K | 1 | 7.220 ms | 7.262 ms | 0.994x | -0.58% |
| 100K | 8 | 7.958 ms | 7.968 ms | 0.999x | -0.12% |
| 100K | 32 | 10.385 ms | 10.248 ms | 1.013x | +1.34% |
| 100K | 64 | 13.798 ms | 12.858 ms | 1.073x | +7.31% |
| 100K | 128 | 17.235 ms | 16.623 ms | 1.037x | +3.68% |
| 100K | 1024 | 74.715 ms | 69.581 ms | 1.074x | +7.38% |
| 200K | 1 | 7.307 ms | 7.651 ms | 0.955x | -4.51% |
| 200K | 8 | 8.183 ms | 8.140 ms | 1.005x | +0.53% |
| 200K | 32 | 11.191 ms | 10.927 ms | 1.024x | +2.42% |
| 200K | 64 | 15.575 ms | 14.098 ms | **1.105x** | **+10.48%** |
| 200K | 128 | 20.513 ms | 19.074 ms | 1.075x | +7.54% |
| 200K | 1024 | 99.351 ms | 89.501 ms | **1.110x** | **+11.01%** |

#### 2026-08-15 correction: the e2e attribution fails Amdahl's-law accounting

At 200K/B1024, the baseline selector share estimate is
`21 * 0.488980 / 99.351 = 10.34%`. The measured 1.724x fixed-GVR selector
speedup can save 4.314 ms, predicting a 95.037-ms forward and 1.045x speedup.
Even a 2x selector gives only 1.054x. The observed 89.501-ms forward saves
9.850 ms, leaving 5.536 ms unexplained; top-k alone would need about a 24.5x
speedup to produce that result. The reported 1.110x must therefore not be
claimed as a causal GVR e2e speedup.

The old-to-fixed comparison is independently inconsistent. At 200K/B1024 the
repeated-capture selector improves from roughly 585 to 284 us, which would
save about 6.3 ms across 21 layers, while the forward changes only 0.286 ms.
At 50K the selector improves by roughly 25 us per layer and the forward is
essentially unchanged. This shows that the duplicated native-B32 capture does
not model the real B1024 GVR admission distribution, while the separate server
runs also contain unrelated graph/kernel, TP-rank-wait, clock, and system
variation.

One possible causal path was checked: different output ordering could alter
the downstream sparse-attention memory pattern even with an identical selected
set. On the repeated 200K/B1024 capture, baseline and GVR selected identical
FP32 value sets, but adjacent indices occupied the same 64-token block 19.3%
and 13.2% of the time respectively. This provides no evidence for a GVR
locality benefit and cannot explain the missing 5.536 ms.

The forward table is retained as raw separate-run data, not as a valid GVR
speedup table. A corrected experiment must profile selector totals on the
actual e2e rows, take the TP critical path instead of rank 0 alone, repeat each
server configuration, and preferably alternate paired baseline/GVR CUDA graphs
inside one warmed process. Until then, only the captured-input kernel results
are defensible.

### Cleanup and validation

The force-all dispatch override and CUDA-event CSV hook were measurement-only
and have been removed. The permanent change is the adaptive CuTe DSL rung
policy plus its exactness test. The focused GVR suite passed:

```text
.venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k gvr -v
7 passed, 177 deselected
```

This covers stable request-indexed hint state, production dispatch, FP32/FP16
cold starts, and exact adaptive-rung results at 50K/100K/200K. Targeted Ruff,
Python compilation, and `git diff --check` also passed. FlashInfer autotuning
was disabled for every fixed-GVR e2e server run.

### Backup snapshot

The implementation, tests, benchmark utilities, documentation, and compact
GSM8K result artifacts were committed to the non-main branch
`backup/gvr-fixed-fp32-20260814`. The source snapshot is commit `1daf3b4c2e`.
The 7.3 GB generated Nsight `prof/` directory was intentionally excluded.

### Speculative-decoding assessment

GVR is currently disabled fail-closed when `speculative_config` is present;
the real-model numbers above are all non-speculative. This is not just an
unmeasured dispatch combination. With `S` speculative tokens, the indexer
produces `S + 1` rows per request, while the GVR state has only one top-k row
per stable request ID. The current call would slice a request-ID buffer of
length `B` as if it had `B * (S + 1)` entries, and expanding each ID naively
would make multiple GVR rows race while writing the same persistent state.

The selector itself remains exact with a stale or unrelated valid hint because
its fallback and final selection are exact. A correct first implementation can
therefore expand request IDs across speculative positions, use the last
accepted-step state as the common hint for all verification rows, disable the
per-row fused state write, and deterministically commit one output row per
request afterward. Committing the row corresponding to the accepted prefix
would preserve the best temporal locality, but requires connecting the state
update to speculative acceptance; committing the last real verification row
is still exact but may give a worse hint after rejection.

Speculation may move the kernel crossover to a smaller request batch because
selector occupancy follows effective rows, approximately `B * (S + 1)`. For
example, five target positions turn B8 into 40 selector rows. The existing
real-capture matrix suggests a directional crossover near 64 effective rows
at 50K/100K and near 32 at 200K, while GVR still loses at 10K even at B1024.
These are not spec-decode measurements: verification rows share a request and
hint, so their admission distribution can differ from repeated independent
decode rows.

An end-to-end speculative gain will also be diluted by draft generation,
acceptance, and sampling work outside the target-model forward. Validation
must compare the same proposer and acceptance trace, report both target-forward
latency and accepted output tokens/s, and check exact top-k under variable
decode lengths and CUDA-graph padding. Until that support and benchmark exist,
no speculative GVR speedup should be claimed.

## Full revalidation: paired, component-accounted experiment

The earlier separate-server end-to-end table failed an Amdahl consistency
check and is being rerun from scratch. The new measurement-only harness
captures two FULL decode CUDA graphs in one warmed process: one forces the
existing FP32 selector and one forces GVR. Every decode step replays both
graphs on the same real-model input buffers and alternates which graph runs
first. It records, on every TP rank, both the complete graph time and the sum
of CUDA-event intervals around all indexer top-k calls. FlashInfer autotuning
remains disabled.

Each result must satisfy the following accounting identity within paired-run
noise:

```text
predicted GVR forward = baseline forward
                      - baseline selector total
                      + GVR selector total
predicted speedup = baseline forward / predicted GVR forward
```

The TP critical path is the maximum rank time, not rank 0 alone. The report
will also split samples by execution order, because a cache/clock/order effect
that changes sign when the order alternates is not a kernel speedup. A claimed
gain larger than the measured selector saving is invalid unless another
specific component is measured and explains the difference.

The instrumentation passed targeted Ruff, formatting, Python compilation, and
the existing GVR correctness suite (`7 passed, 177 deselected`). A standalone
CUDA-graph probe also verified that preallocated external timing events can be
re-recorded during capture and read after replay; creating ordinary timing
events inside capture was rejected by CUDA and was not used for model data.

The first real-checkpoint smoke run was discarded before analysis. Although
the API process started from this checkout, its spawned TP workers resolved a
stale editable vLLM install at `/home/woosuk/workspace-agents/vllm-3`; this was
confirmed by worker traceback paths, a graph-capture log line that mapped only
to that checkout, and the absence of all paired CSVs. The environment was
reinstalled with
`VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`; the installed
version then changed from the stale `gc79475406` source to this branch's
`gefbfdd3df` source. No latency from the discarded run is used below.

The first no-internal-selector-events control also produced no measurements:
the globally enabled Rust frontend exited at its fixed 600-second engine
registration timeout while the 432.9-GiB checkpoint load and paired graph
capture were still in progress. The model workers had not failed, but their
parent then terminated them. The control was restarted with
`VLLM_USE_RUST_FRONTEND=0`, using the in-process Python frontend so startup is
not bounded by that registration timer. This changes only frontend startup;
the TP4 model, weights, input workload, CUDA graphs, and GPU-event forward
measurement are unchanged.

The standalone CUDA-graph matrix has now been rerun with the repaired local
extension and real captures. Native B1/B8/B32 GVR speedups range from 0.519x
to 1.190x; B64/B128/B1024 repeated-B32 scaling reaches 2.122x, 1.993x, and
1.722x at 200K. The native 100K FP16 baseline improves by only 1.021x,
1.020x, and 1.030x at B1/B8/B32; repeated-row B64/B128/B1024 improves by
1.119x, 1.123x, and 1.210x. FP16 versus FP32 retains 99.9714% of selected
indices but changes the exact set on 40/82 rows, so this is selector overlap,
not an end-to-end model-accuracy result.

The component-accounted paired run is also complete. All retained samples
contain exactly 21 baseline and 21 GVR selector intervals. In 23/24 matrix
cells, the observed full-forward delta differs from the measured selector
delta by under 0.1 ms, validating the Amdahl accounting. Crucially, the actual
199.8K/B1024 rows measure 10.973 ms baseline selector total versus 12.879 ms
for GVR: the real B1024 hint/admission distribution reverses the 1.722x win on
the repeated-B32 microbenchmark. A no-internal-events paired control is in
progress to remove the 42 timing nodes from the final absolute e2e latencies.

The first no-events workload pass produced all requested requests, but its
driver serialized the same large token-ID JSON separately in every coroutine.
At 100K/199.8K this delayed later admissions long enough that early requests
completed before the full B1024 wave was resident; several long-context B128
runs were shortened for the same reason. Those partial rows are excluded. The
driver now serializes the identical temperature-zero payload once per wave and
reuses the bytes for every HTTP request. This removes client CPU serialization
from admission without changing prompts, model execution, or the GPU-event
latency metric. Only the affected exact-batch cells are being rerun.

The no-events control and targeted admission reruns are complete. Every one of
the 24 cells now has 56-230 retained exact-batch samples after trimming, with
equal baseline-first and GVR-first counts. The order-balanced real-forward
speedup range is only 0.9968-1.0016x. The largest apparent win is 0.159% at
50K/B128, the largest loss is 0.323% at 100K/B1, and 199.8K/B1024 is
0.9997x (99.675 versus 99.706 ms), not the old 1.110x claim.

The component events were causal perturbations, not passive measurements. At
10K/B1 they add 0.143 ms to baseline but 0.594 ms to GVR; that 0.451-ms
differential explains almost the entire instrumented 0.459-ms loss. At
199.8K/B1024 they add 14.857 and 17.350 ms respectively; the 2.493-ms
differential explains almost the entire instrumented 2.524-ms loss. The
selector's summed event duration is therefore not the serial Amdahl fraction:
the events impose ordering and suppress overlap. The final report uses only
the no-internal-events graph times for real e2e conclusions.

All paired measurement hooks have now been removed from production model code.
The reusable real-data driver keeps the admission fix: it serializes one
temperature-zero request body per wave and shares those immutable bytes across
the concurrent posts. This is necessary to reach the requested large batch
before early requests complete and does not affect GPU timing.

Final validation, using only `/home/woosuk/workspace/vllm/.venv`, passed all
15 selected GVR and FP32/FP16 padded-stride cases (`169 deselected`). Targeted
Ruff check/format, Python compilation, and `git diff --check` also passed.

## Event-free selector attribution correction

The explanation that internal selector timing events "suppress overlap" is
being retracted and rechecked. The GLM-5.2 portable DeepSeek-V3.2 selector runs
on the ambient CUDA stream; unlike the DeepSeek-V4 implementation, this path
does not explicitly schedule indexer work on auxiliary streams. Inserting
start/end event-record nodes around an already ordered single-stream selector
therefore adds graph work, but does not by itself remove a demonstrated
selector/neighbor overlap.

CUDA does support pre-created timing events as explicit event-record nodes in
a graph. A local `.venv` probe on GB200 successfully replayed and timed such a
graph. A second single-stream probe measured 26.67 us for 21 tiny kernel nodes
and 208.30 us after adding 42 external timing-event nodes. This establishes
that the nodes are compatible and non-passive; it does not explain the much
larger or implementation-dependent deltas in the previous model runs.

The event-instrumented and event-free model results were separate captures, so
subtracting their absolute latencies does not prove that the difference was
caused by lost overlap. The recorded latencies remain observations, but it was
not yet established that the nominal GVR graph actually dispatched GVR. The
next check will attribute selector
kernels from an event-free Nsight Systems/CUPTI trace, verify graph streams and
dependencies, and apply Amdahl's Law only to work shown to lie on the serial
critical path.

The first replacement capture exposed a separate dispatch mistake before its
numbers were used. With `max_model_len=200032`, the decode logits width was
200,032, which fails GVR's `num_columns % 64 == 0` eligibility condition. The
nominally forced trace consequently contained the baseline selector plus 21
GVR state-store launches per replay, not the GVR selector. That capture is
discarded. The controlled event-free pair is being repeated with
`max_model_len=200000`, which covers the 199,800-token prompt plus 32 generated
tokens and is a valid GVR width.

## Final event-free attribution result

The corrected capture is complete and overturns both the overlap explanation
and the neutral-GVR conclusion. Nsight Systems recorded CUDA and NVTX activity
with `--cuda-graph-trace=node`; no selector timing events were inserted into
the graphs. Both servers used the real GLM-5.2-NVFP4 checkpoint, TP4, the
portable DeepSeek-V3.2 implementation, full-decode CUDA graphs at B1/B1024,
20 GiB KV cache per rank, and disabled FlashInfer autotuning. The only selector
difference was baseline versus forced GVR.

To obtain a real fixed B1024 plateau, the driver now supports
`--single-request-n`. One real document prompt creates 1,024 child sequences in
one completion request. A 199,400-token prompt with 512 requested output tokens
keeps the children alive through admission; analysis retains only NVTX ranges
explicitly labeled `execute_context_0(0)_generation_1024(1024)`. The baseline
and GVR traces contain 130 and 146 analyzed exact replays per rank,
respectively.

Dispatch validation passed. Every retained baseline replay contains exactly 21
`FilteredTopKUnifiedKernel` launches and no GVR kernel. Every retained GVR
replay contains exactly 21 fused CuTe GVR launches and no cooperative or
filtered selector. Both selectors execute on stream 19. Intersecting every
selector interval with every kernel interval on all other streams finds zero
overlap in both traces.

| Case | Baseline forward | GVR forward | Baseline selector | GVR selector | Baseline share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 199.8K/B1 | 7.615 ms | 7.510 ms | 0.322 ms | 0.341 ms | 4.24% |
| 10K/B1024 | 52.709 ms | 52.030 ms | 1.168 ms | 0.960 ms | 2.21% |
| 199.4K/B1024 | 99.828 ms | 89.857 ms | 12.662 ms | 2.785 ms | 12.68% |

At 199.4K/B1024, GVR makes the selector 4.55x faster and saves 9.876 ms.
Amdahl's Law predicts a complete forward of
`99.828 - 12.662 + 2.785 = 89.951 ms`, or 1.1098x. The measured forward is
89.857 ms, or **1.1110x** (11.10% fixed-batch throughput improvement and 9.99%
latency reduction). The 0.096-ms residual is consistent with the measured
non-selector differences: indexer logits are 0.290 ms slower under GVR while
sparse attention is 0.194 ms faster.

### Why the standalone B1024 selector reports only 1.722x

The standalone row is a valid timing of its constructed input, but calling it a
representative B1024 model result is wrong. The offline dataset retained only
the first sparse-indexer layer at two consecutive native-B32 decode steps
(`call0` and `call21`). Later layer captures were invalid. Its nominal B1024
input repeats those 32 rows 32 times; it does not sample the other 20 selector
layers.

Per-call attribution of the independent BEAM trace isolates the gap. These are
medians for each call ordinal over all exact-B1024 graph replays and TP ranks:

| Selector ordinal | Model layer | Baseline | GVR | Baseline / GVR |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 511.568 us | 293.696 us | 1.742x |
| 2 | 1 | 542.720 us | 485.040 us | 1.119x |
| 3 | 2 | 613.888 us | 1,213.680 us | 0.506x |
| 4 | 6 | 614.080 us | 44.576 us | 13.776x |
| 5 | 10 | 613.216 us | 44.416 us | 13.806x |
| 6 | 14 | 612.224 us | 44.432 us | 13.779x |
| 7 | 18 | 611.904 us | 44.384 us | 13.787x |
| 8 | 22 | 611.936 us | 44.256 us | 13.827x |
| 9 | 26 | 612.288 us | 44.544 us | 13.746x |
| 10 | 30 | 612.432 us | 44.576 us | 13.739x |
| 11 | 34 | 612.256 us | 44.416 us | 13.785x |
| 12 | 38 | 612.000 us | 44.384 us | 13.789x |
| 13 | 42 | 612.416 us | 44.544 us | 13.749x |
| 14 | 46 | 612.224 us | 44.544 us | 13.744x |
| 15 | 50 | 612.160 us | 44.384 us | 13.792x |
| 16 | 54 | 612.256 us | 44.384 us | 13.795x |
| 17 | 58 | 612.160 us | 44.480 us | 13.763x |
| 18 | 62 | 612.240 us | 44.448 us | 13.774x |
| 19 | 66 | 612.016 us | 44.544 us | 13.740x |
| 20 | 70 | 612.176 us | 44.352 us | 13.803x |
| 21 | 74 | 612.256 us | 44.480 us | 13.765x |
| **Sum of ordinal medians** | -- | **12.690 ms** | **2.793 ms** | **4.544x** |

Every ordinal has 384 direct measurements. Selectors 4--21 individually span
13.739x--13.827x; their consistency is measured, not extrapolated from
selector 4.

### Paper-alignment audit: the 13.8x later-layer result is not credible

Comparing the table with the
[GVR paper](https://arxiv.org/abs/2604.22312) uncovered a first-principles
failure. The paper profiles real DeepSeek-V3.2 logits at about 70.7K tokens and
reports 1.59x--2.04x average per-layer speedup, 1.88x overall, and 2.42x as the
largest sampled per-layer/per-step speedup. Its early layers 0--1 have only
about 1--2% previous-step top-k overlap and require 2.1--2.7 threshold-search
iterations; layers 20--60 have about 35--50% overlap and require 1.1--1.6
iterations. Thus weaker early layers and faster deeper layers are qualitatively
expected. An abrupt 13.8x plateau is not.

The 44.5-us number also violates the memory-traffic lower bound. A nominal
B1024-by-199,400 FP32 selector input contains:

```text
1024 * 199400 * 4 = 816,742,400 bytes
816,742,400 / 44.5 us = 18.4 TB/s
```

That is the bandwidth required for only one read. The local GB200 has about
8 TB/s HBM bandwidth, while this GVR implementation requires at least a rung
count scan and a candidate-collection scan, plus hint, refinement, and output
traffic. Score distribution can change the number of refinement passes, but
it cannot remove the mandatory full-row scans.

A direct CUDA-graph diagnostic used the same B1024-by-W200K FP32 launch shape
and changed only the runtime `seq_lens`:

| Runtime sequence length | GVR latency |
| ---: | ---: |
| 1 | 8.879 us |
| 2,048 | 8.358 us |
| 4,096 | 41.039 us |
| 8,192 | 47.235 us |
| 16,384 | 66.343 us |
| 32,768 | 87.782 us |

The observed 44--45 us for selectors 4--21 matches a 4K--8K effective-row
path, not a 199.4K full-row path. Because all indexer layers in one forward
must receive the same sequence lengths, temporal overlap or a favorable layer
distribution cannot explain this transition. The leading hypotheses are that
selectors 4--21 see an incorrect runtime length, that an earlier kernel
corrupts metadata, or that another unintended early-exit condition is active.
The Nsight trace records launch shape but not argument values, so it cannot
identify which hypothesis is correct.

The [IndexCache paper](https://arxiv.org/abs/2603.12201) supports only the
broader observation that layers are heterogeneous. It studies cross-layer
index reuse, not GVR's same-layer temporal reuse: adjacent layers show
70--100% top-k overlap, with distinct clusters and sensitive early/transition
layers. That explains why GLM can retain a nonuniform subset of indexer layers;
it does not predict a 13.8x top-k kernel speedup for the retained layers.

The 4.55x selector ratio and 1.1110x forward ratio are therefore quarantined.
Amdahl's Law shows that the measured forward difference equals the measured
selector difference, but an incorrect early exit would satisfy the same
identity. Before using either number, native inputs from selectors 4--21 must
be checked against exact top-k, their runtime sequence lengths must be read
back, and a profiler must confirm full-row memory traffic.

The standalone 487.619/283.104-us result is within 5% of the traced first call
and gives nearly the same 1.72--1.74x ratio. It never measured an average
layer. The traced 4.55x ratio is dominated by the now-quarantined calls 4--21.
The original document-prompt trace reproduces the same ordinal pattern, but
reproducing a likely wrong-length path does not validate it.

All 21 exact-B1024 GVR calls use the same compiled launch configuration: grid
1024, block size 512, 64 registers per thread, and 60,140 bytes of dynamic
shared memory. The model also owns separate GVR state buffers for every hidden
layer. Those facts rule out launch tuning as the source of the timing cliff,
but they do not show that each call saw the same runtime `seq_lens` or executed
the normal full-row path.

The trace proves the aggregate timing ratio and its accounting connection to
the graph latency; it does not prove exact selector behavior. The next
diagnostic must inspect runtime lengths and value-set correctness, not merely
rung choice or candidate counts.

The B1 whole-forward ratio must not be attributed to GVR. Its selector is
about 19 us slower than the cooperative baseline, while unrelated kernels vary
between the separate servers. This supports retaining the production small-row
guard.

The prior event-free 0.9997x result did not execute GVR. Its 200,032-column
shape failed the `% 64 == 0` condition, and the replacement trace shows the
baseline selector plus state storage. The internal-event experiment did add 42
event-record nodes, but its earlier "suppressed overlap" explanation is false:
the uninstrumented selector is already serialized. Event overhead and the
dispatch error were separate problems.

Reports and SQLite exports are:

- `/tmp/gvr_eventfree_baseline_n200000_20260815.nsys-rep` and `.sqlite`;
- `/tmp/gvr_eventfree_gvr_n200000_20260815.nsys-rep` and `.sqlite` for B1 and
  10K/B1024;
- `/tmp/gvr_eventfree_gvr_long_n200000_20260815.nsys-rep` and `.sqlite` for
  the exact 199.4K/B1024 plateau.

Final validation used only this checkout's `.venv`: all 15 selected GVR and
padded-stride tests passed (`169 deselected`), as did Ruff check/format, Python
compilation, typos, `GVR_SUMMARY.md` Markdown lint, and `git diff --check`.
Whole-file Markdown lint for the two historical long-form documents still
reports their pre-existing MD060 table-spacing errors.

## Independent BEAM-corpus revalidation

> **Quarantined by the later paper-alignment audit above.** This section
> reproduces the same timing path on independent input, but the 44-us calls
> fail the full-row bandwidth bound. Reproduction is not correctness
> validation.

The 4.546x real-model selector ratio was independently revalidated with the
[Kimi Vendor Verifier BEAM corpus][beam]
instead of the repository-document prompt used by the first event-free trace.
The source checkout is commit `3dad65a760a8867cda72f6dd8848d876a4e851b4`.
The downloaded 1M-chat and question objects match their Git-LFS SHA-256 values
`878eefe2b90273b52ad1b0ddd29e72568a6b218567c0a8cdba78696099cba2ad`
and `0882ca1ea1dfffa4645463b94c44156277e4da082fb9a1a68ceeca852b4fc2f4`.

The test prompt uses BEAM chat 1 and its first abstention probe. It applies the
served GLM-5.2 tokenizer's chat template to 592 real conversation messages and
the official probing prefix, then left-truncates to exactly 199,400 tokens to
fit the 200,000-token deployment. The resulting token-ID artifact has SHA-256
`9751fea511c7b08448da6b2b58ea50ad8d3c28244364b64e2f0c90e89b0a5ff7`.
FlashInfer autotuning remained disabled. Both runs used the same checkpoint,
TP4 configuration, full-decode CUDA graph, prompt IDs, sampling seed, and
`temperature=1.0`. Nsight Systems captured CUDA and NVTX activity without
inserting timing events into the model graph. Analysis retained only ranges
labeled `execute_context_0(0)_generation_1024(1024)`.

The first baseline profiling range was discarded before inference. The API
correctly rejected `n=1024` with greedy `temperature=0` sampling, so it contains
no B1024 model forwards. The rerun uses BEAM's default `temperature=1.0`, which
is accepted for multi-output generation and produces independently sampled
child trajectories. The prefix-only B1 warmup completed in 17.232 seconds and
was outside the discarded profiler range.

The valid pair contains 384 exact-B1024 replays per selector: 96 decode steps
on each of four TP ranks. Every replay maps to one CUDA graph launch containing
2,043 kernels. Every baseline replay has exactly 21
`FilteredTopKUnifiedKernel` launches and no GVR kernel; every GVR replay has
exactly 21 fused GVR launches and no filtered selector.

| Median event-free graph metric | Baseline | GVR | Baseline / GVR |
| --- | ---: | ---: | ---: |
| One TP-rank forward | 99.757 ms | 89.639 ms | 1.1129x |
| 21 selectors in one TP-rank forward | 12.709 ms | 2.796 ms | 4.5456x |
| 21-call selector total divided by 21 | 605.204 us | 133.141 us | 4.5456x |
| TP critical-path forward | 99.814 ms | 89.682 ms | 1.1130x |

The original repository-document trace reported aggregate/21 values of
602.939 us versus 132.639 us, or 4.546x. These are arithmetic normalizations,
not representative single calls. The independent BEAM data changes the
aggregate/21 values by less than 0.4% and reproduces the ratio to three decimal
places. The individual-rank forward distribution is also narrow: the
baseline/GVR 5th-to-95th percentile
ranges are 99.381--100.185 ms and 89.387--90.006 ms.

Using the median per-step TP-rank mean, baseline selector share is 12.74% and
the measured selector totals are 12.716 ms versus 2.795 ms. Amdahl's Law then
predicts `99.814 / (99.814 - 12.716 + 2.795) = 1.1104x`; the directly observed
critical-path result is 1.1130x. The predicted forward is 89.993 ms, only
0.311 ms slower than the measured 89.682 ms. Thus the trace attributes the
graph difference to the selector path. It does not establish that the path
performed exact full-length selection; the 11.3% whole-forward difference is
quarantined with the selector result.

The valid reports and exports are
`/tmp/gvr_beam_baseline_valid_20260815.{nsys-rep,sqlite}` and
`/tmp/gvr_beam_gvr_valid_20260815.{nsys-rep,sqlite}`. This is one real BEAM
conversation and probe, not a corpus-wide model-quality evaluation; its 96
exact-B1024 decode steps per rank test the performance claim rather than model
accuracy.

[beam]: https://github.com/MoonshotAI/Kimi-Vendor-Verifier/tree/main/beam

## Native 21st-selector microbenchmark

> **Quarantined by the paper-alignment audit.** The timing is directly
> observed, but its 44-us duration is consistent with only a 4K--8K effective
> row and cannot represent the claimed 199.4K full-row execution.

The requested kernel time does not require adding events or capture nodes to
the CUDA graph. The normal BEAM model traces above are event-free Nsight
CUDA/NVTX captures. Sorting the 21 selector kernels on their shared stream 19
within each exact-B1024 graph replay gives the following 21st-call result over
384 samples (96 decode steps on each of four TP ranks):

| 21st selector kernel | Median | Mean | P05--P95 |
| --- | ---: | ---: | ---: |
| Baseline | 612.256 us | 611.991 us | 608.256--614.496 us |
| GVR | 44.480 us | 44.544 us | 43.904--45.408 us |

The directly observed timing ratio is 13.765x, but it is not a valid full-row
speedup until runtime length and output correctness are verified. Nsight
reports one stable configuration for every retained 21st call: baseline uses
grid/block 1024/1024 with 60 registers and 131,072 bytes dynamic shared memory;
GVR uses grid/block 1024/512 with 64 registers and 60,140 bytes dynamic shared
memory.

A temporary tensor-copy experiment was also attempted to feed the existing
standalone replay. Copying a full `(1024, 200000)` layer-74 tensor out of the
worker required eager execution, so it changed the execution mode being
investigated. Its timings are not used to reinterpret the normal CUDA-graph
trace, and the temporary source hook was removed. No CUDA event or tensor-copy
node was added to the production graph.

## Correctness investigation of the 44-us anomaly

The saved real BEAM layer-74 input provides a direct correctness check at the
claimed shape. It contains FP32 logits of shape `(1024, 200000)`, hint indices
of shape `(1024, 2048)`, and per-row sequence lengths from 199,450 to 199,481.
On this input, the current GVR kernel returned no invalid indices and every
row's selected value multiset exactly matched `torch.topk` after masking each
row at its sequence length. Its CUDA-graph replay time was 668.242 us, not
44.5 us.

The production fused hint/state path was checked separately on the same
capture. It also had zero incorrect rows or values, did not modify the input
sequence-length tensor, and stored output/state-valid data exactly as expected.
This rules out both the core selector and fused state update as the source of
the anomalous timing.

As a length-sensitivity control, the same `(1024, 200000)` allocation and
launch shape took 8.358 us at runtime length 2,048, 41.039 us at 4,096,
47.235 us at 8,192, and 668.242 us on the real approximately-199.45K rows.
This initially made the traced 44.5-us calls look like a roughly 4K--8K
effective runtime length, but duration alone was not unique evidence for that
explanation.

The subsequent reproducer found a different cause: the old kernel treated an
unusable hint bracket (`hint_min >= hint_max`, including repeated gathered
values) as proof that the complete row was degenerate. It immediately emitted
indices `[0, 2048)` without scanning logits. With random logits and temporal
hints containing only index zero, all 64/64 rows disagreed with exact top-k and
the kernel took only 64.422 us at `(64, 65536)`. This shortcut, rather than a
short runtime sequence length, explains how a nominal full-width graph node
could take tens of microseconds.

The fix first rebuilds the bracket from evenly spaced cold hints. If those
values are also degenerate, it scans the complete row for global bounds;
identity output is now allowed only when that scan proves the entire row has
one value. Both the ordinary repeated-hint reproducer and an adversarial case
whose temporal and evenly spaced hints all gather zero now match exact
`torch.topk` on every row. A regression test covers the adversarial path.

### Corrected real-model result

An event-free Nsight trace repeated the real BEAM experiment after the fix.
It used the Python frontend, the same 199,400-token prompt, GLM-5.2-NVFP4
weights, TP4, 20-GiB KV cache per rank, full-decode CUDA graphs captured at B1
and B1024, FP32 indexer logits, and disabled FlashInfer autotuning. There are
384 exact-B1024 graph replays, and every replay contains 2,043 kernels and 21
GVR selectors. Correlation IDs were paired with process IDs because CUPTI
correlation IDs are only process-local.

| Selector | Layer | Baseline | Fixed GVR | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 511.568 us | 296.352 us | 1.726x |
| 2 | 1 | 542.720 us | 488.448 us | 1.111x |
| 3 | 2 | 613.888 us | 1,211.968 us | 0.507x |
| 4 | 6 | 614.080 us | 313.856 us | 1.957x |
| 5 | 10 | 613.216 us | 312.384 us | 1.963x |
| 6 | 14 | 612.224 us | 311.712 us | 1.964x |
| 7 | 18 | 611.904 us | 312.240 us | 1.960x |
| 8 | 22 | 611.936 us | 311.680 us | 1.963x |
| 9 | 26 | 612.288 us | 312.384 us | 1.960x |
| 10 | 30 | 612.432 us | 312.400 us | 1.960x |
| 11 | 34 | 612.256 us | 311.680 us | 1.964x |
| 12 | 38 | 612.000 us | 311.552 us | 1.964x |
| 13 | 42 | 612.416 us | 312.320 us | 1.961x |
| 14 | 46 | 612.224 us | 312.304 us | 1.960x |
| 15 | 50 | 612.160 us | 311.776 us | 1.963x |
| 16 | 54 | 612.256 us | 312.224 us | 1.961x |
| 17 | 58 | 612.160 us | 311.728 us | 1.964x |
| 18 | 62 | 612.240 us | 311.663 us | 1.964x |
| 19 | 66 | 612.016 us | 312.288 us | 1.960x |
| 20 | 70 | 612.176 us | 311.664 us | 1.964x |
| 21 | 74 | 612.256 us | 312.256 us | 1.961x |

The median per-replay selector totals are 12.709 ms for baseline and
7.621 ms for fixed GVR: **1.668x**, a 40.04% selector-time reduction. The
median per-rank graph spans are 99.757 ms and 94.557 ms: **1.0550x** end-to-end
throughput, or 5.21% forward-latency reduction. Baseline selector share is
12.74%. Amdahl substitution predicts
`99.757 - 12.709 + 7.621 = 94.668 ms` (1.0537x); the observed fixed forward is
only 0.112 ms faster. The corrected result is therefore first-principles
consistent. It supersedes the invalid 4.55x selector and 1.111x end-to-end
claims produced by the unchecked identity shortcut.

Correctness was verified at three levels: the adversarial regression now
matches exact `torch.topk` for FP32 and FP16; all nine selected GVR tests pass; and the saved
real layer-74 `(1024, 200000)` input has zero invalid indices and zero
mismatched rows after the fix, with a 668.662-us standalone graph time. The
fixed model trace is
`/tmp/gvr_beam_gvr_fixed_py_20260815.{nsys-rep,sqlite}`.

Final repository checks passed Ruff check/format, typos, mypy, SPDX headers,
the CUDA-API guard, and `git diff --check`. Whole-file Markdown lint still
reports the historical MD060 compact-table errors in `GVR.md`; all reported
lines predate this section (the last is line 1,535 before these additions).

## 2026-08-15 fixed-kernel full performance rerun

This rerun measures the corrected degenerate-hint recovery across the complete
requested matrix: KV lengths 10K, 50K, 100K, and approximately 200K, and
batches 1, 8, 32, 64, 128, and 1024. The standalone selector sweep compares
FP32 and FP16 GVR with the matching production selector under CUDA-graph
replay and checks every GVR result against exact `torch.topk`. Native real
captures are available at B1/B8/B32; larger standalone batches repeat real B32
rows and are therefore throughput-scaling controls, not substitutes for real
large-batch distributions. The end-to-end sweep uses real GLM-5.2-NVFP4
weights and prompt tokens, TP4, full decode CUDA graphs, and disabled
FlashInfer autotuning.

### Standalone selector microbenchmark

Each cell is the median CUDA-graph replay time across 50 saved model nodes and
50 timed replays per node. All fixed-GVR outputs matched exact `torch.topk`;
there were zero mismatched rows. Times are in microseconds and speedup is
baseline divided by fixed GVR.

#### FP32 selector

| KV length | Batch | Baseline (us) | Fixed GVR (us) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10K | 1 | 5.057 | 9.771 | 0.518x |
| 10K | 8 | 5.204 | 9.947 | 0.523x |
| 10K | 32 | 5.333 | 9.971 | 0.535x |
| 10K | 64 | 5.521 | 10.252 | 0.539x |
| 10K | 128 | 5.539 | 9.910 | 0.559x |
| 10K | 1024 | 34.085 | 43.182 | 0.789x |
| 50K | 1 | 10.389 | 16.519 | 0.629x |
| 50K | 8 | 10.536 | 18.325 | 0.575x |
| 50K | 32 | 13.208 | 18.095 | 0.730x |
| 50K | 64 | 24.459 | 18.954 | 1.290x |
| 50K | 128 | 24.657 | 18.439 | 1.337x |
| 50K | 1024 | 171.777 | 98.240 | 1.749x |
| 100K | 1 | 11.108 | 14.524 | 0.765x |
| 100K | 8 | 11.574 | 20.929 | 0.553x |
| 100K | 32 | 14.610 | 20.671 | 0.707x |
| 100K | 64 | 29.758 | 23.093 | 1.289x |
| 100K | 128 | 30.029 | 25.382 | 1.183x |
| 100K | 1024 | 219.159 | 139.631 | 1.570x |
| 200K | 1 | 12.611 | 18.359 | 0.687x |
| 200K | 8 | 15.098 | 19.883 | 0.759x |
| 200K | 32 | 24.323 | 20.479 | 1.188x |
| 200K | 64 | 54.789 | 26.194 | 2.092x |
| 200K | 128 | 67.190 | 34.116 | 1.969x |
| 200K | 1024 | 487.606 | 281.493 | 1.732x |

The FP32 crossover depends strongly on sequence length. Fixed GVR does not win
at 10K in this matrix; it crosses over at B64 for 50K and 100K, and at B32 for
200K. This is consistent with fixed launch/refinement overhead dominating when
the baseline has little input to scan.

#### FP16 selector

| KV length | Batch | Baseline (us) | Fixed GVR (us) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10K | 1 | 9.905 | 9.000 | 1.101x |
| 10K | 8 | 8.987 | 9.195 | 0.977x |
| 10K | 32 | 9.737 | 9.380 | 1.038x |
| 10K | 64 | 8.539 | 9.772 | 0.874x |
| 10K | 128 | 8.593 | 9.400 | 0.914x |
| 10K | 1024 | 55.321 | 33.657 | 1.644x |
| 50K | 1 | 10.347 | 20.807 | 0.497x |
| 50K | 8 | 10.402 | 20.609 | 0.505x |
| 50K | 32 | 12.951 | 20.395 | 0.635x |
| 50K | 64 | 21.258 | 21.047 | 1.010x |
| 50K | 128 | 21.454 | 20.082 | 1.068x |
| 50K | 1024 | 146.028 | 94.132 | 1.551x |
| 100K | 1 | 10.756 | 14.105 | 0.763x |
| 100K | 8 | 11.300 | 19.806 | 0.571x |
| 100K | 32 | 14.094 | 19.687 | 0.716x |
| 100K | 64 | 26.533 | 21.559 | 1.231x |
| 100K | 128 | 26.733 | 23.539 | 1.136x |
| 100K | 1024 | 181.450 | 100.657 | 1.803x |
| 200K | 1 | 12.576 | 26.648 | 0.472x |
| 200K | 8 | 14.688 | 29.475 | 0.498x |
| 200K | 32 | 23.096 | 30.092 | 0.768x |
| 200K | 64 | 48.256 | 38.965 | 1.238x |
| 200K | 128 | 48.493 | 48.529 | 0.999x |
| 200K | 1024 | 335.737 | 364.632 | 0.921x |

FP16 is not a monotonic improvement after the correctness fix. Quantization
can collapse distinct hint values into a degenerate bracket; the corrected
kernel must then retry cold hints and may perform a full-row bounds recovery.
The invalid shortcut previously hid that work. FP16 therefore needs a
separate measured dispatch policy and must not be assumed faster than FP32.

### End-to-end model forward latency

This sweep forced FP32 GVR at every batch size only for measurement; the
production `num_rows >= 32` dispatch was restored afterward. The table reports
median per-rank CUDA-graph spans in milliseconds. B1--B128 use one request with
the requested number of decode candidates. A single B1024 request was rejected
because scheduler preemption prevented a sustained exact-B1024 plateau at long
contexts. The replacement uses 1,024 concurrent requests and retains only the
exact-B1024 plateau; baseline and GVR have identical sample counts per context.

| KV length | Batch | Baseline (ms) | Fixed GVR (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10K | 1 | 7.189 | 7.349 | 0.9782x |
| 10K | 8 | 7.751 | 7.878 | 0.9839x |
| 10K | 32 | 9.630 | 9.803 | 0.9823x |
| 10K | 64 | 11.633 | 11.810 | 0.9850x |
| 10K | 128 | 14.364 | 14.431 | 0.9954x |
| 10K | 1024 | 52.420 | 51.678 | 1.0143x |
| 50K | 1 | 7.297 | 7.420 | 0.9834x |
| 50K | 8 | 7.876 | 8.087 | 0.9739x |
| 50K | 32 | 10.015 | 10.119 | 0.9898x |
| 50K | 64 | 12.603 | 12.372 | 1.0187x |
| 50K | 128 | 15.870 | 15.518 | 1.0227x |
| 50K | 1024 | 63.039 | 60.055 | 1.0497x |
| 100K | 1 | 7.351 | 7.638 | 0.9624x |
| 100K | 8 | 8.025 | 8.302 | 0.9666x |
| 100K | 32 | 10.549 | 10.659 | 0.9897x |
| 100K | 64 | 13.527 | 13.134 | 1.0299x |
| 100K | 128 | 17.257 | 16.847 | 1.0243x |
| 100K | 1024 | 74.596 | 73.234 | 1.0186x |
| 199.4K | 1 | 7.437 | 7.966 | 0.9335x |
| 199.4K | 8 | 8.218 | 8.588 | 0.9569x |
| 199.4K | 32 | 11.237 | 11.675 | 0.9625x |
| 199.4K | 64 | 15.204 | 14.661 | 1.0371x |
| 199.4K | 128 | 20.520 | 19.714 | 1.0409x |
| 199.4K | 1024 | 99.333 | 93.613 | 1.0611x |

At B1024, the retained rank-event counts for 10K, 50K, 100K, and 199.4K are
464, 464, 420, and 400 for both implementations. The baseline/GVR median and
5th--95th percentile ranges are respectively 52.420/51.678 ms
(52.334--52.519/51.608--51.798), 63.039/60.055 ms
(62.926--63.261/59.942--60.251), 74.596/73.234 ms
(74.468--74.847/73.064--73.463), and 99.333/93.613 ms
(99.127--99.668/93.461--93.929).

### Interpretation and Amdahl check

The qualitative result is clear: forcing GVR at small batches regresses the
forward pass, while long-context B64 and above generally benefit. The raw
matrix should not be read as proving every sub-percent difference. Baseline
and GVR are separate server runs, so graph-level medians include run-to-run
variation outside the selector.

For example, the 10K B1024 standalone selector is 9.097 us slower per call.
With 21 selector calls, Amdahl substitution predicts a 0.36% forward
regression, not the raw 1.43% improvement. That cell is explicitly not a
causal GVR win. The 50K and 199.4K B1024 raw gains also exceed the savings
predicted by the repeated-B32-row microbenchmark by about 1.4 ms. At 100K
B1024, predicted and measured improvements differ by only 0.31 ms.

The strongest causal large-batch check remains the corrected native real
199.4K B1024 BEAM trace above: its 21 actual selector inputs give a 1.668x
selector speedup and Amdahl predicts 1.0537x end-to-end versus 1.0550x
observed. The new 1.0611x matrix cell is directionally consistent but includes
approximately 0.6% favorable cross-run variation. Thus the defensible real
model result at B1024/200K is about **5.5% throughput improvement**, not an
unqualified 6.1%.

### Artifacts

- Standalone logs: `/tmp/gvr_fixed_micro_b1_20260815.log`,
  `/tmp/gvr_fixed_micro_b8_20260815.log`, and
  `/tmp/gvr_fixed_micro_b32plus_20260815.log`.
- B1--B128 traces: `/tmp/gvr_fixed_matrix_baseline_20260815.nsys-rep` and
  `/tmp/gvr_fixed_matrix_gvr_20260815.nsys-rep` (matching `.sqlite` exports).
- Exact-B1024 traces:
  `/tmp/gvr_fixed_b1024_baseline_concurrent_40g_20260815.nsys-rep` and
  `/tmp/gvr_fixed_b1024_gvr_40g_20260815.nsys-rep` (matching `.sqlite`
  exports).
- Driver logs use the same stems with `_driver_20260815.log`; prompt tokens are
  in `/tmp/gvr_beam_prompt_199400.json`.
