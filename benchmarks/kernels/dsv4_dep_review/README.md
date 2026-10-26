# DeepSeek V4 DEP performance review

This bundle records the frozen MI355X stack selected for human review on
2026-10-26. Runtime code matches the measured `9db2dd585bd9` source diff against `df6603f0583375f244d1ac6d36265462b7490392`.
`source_manifest.json` records the original per-file hashes and the review-only
adaptations. The added report and
configuration documentation do not change runtime behavior. Five synchronization
calls in a graph test use the required `torch.accelerator` API; test assertions
and tolerances are unchanged.

## Scope and provenance

This is an integration draft in Fangzhou-Ai/vllm. Existing upstream proposals
retain ownership of their changes; this draft is not a replacement upstream PR.

| Work | Provenance / local adaptation |
| --- | --- |
| Dense paged MXFP4 indexer | [#60834](https://github.com/vllm-project/vllm/pull/60834), applied commit `022b43c66f6c2b39c8f0e7c4813544481aca43c0`; real operator tests and 64-head coverage added locally |
| DP prefill delayer | [#61834](https://github.com/vllm-project/vllm/pull/61834), head `27661316f53f26eb36b0980c8b5325bbc64141a0`; retained local cadence initialization and real Scheduler/AsyncScheduler tests |
| ASM sparse MLA / native KV layout | [#61826](https://github.com/vllm-project/vllm/pull/61826), head `d36f02e274ae5cc395e37cf6d5e45cacf8152dce`; local planner accounting, minimum split correction, fast compressed decode and short-row correction |
| WO_A FP8 BMM | One-time weight preshuffle and the matching AITER BMM, plus shared BF16 RoPE tables |
| Router | Reuse the existing top-6 Triton router on gfx950 for supported shapes; stable softplus, deterministic ties, real padding mask, zero-row return |
| E8M0 linears | Opt-in native scaled shared-expert, attention and draft GEMMs; exact checkpoint scale conversion; fused query RMSNorm/quantization |
| Indexer TopK | Select native AITER TopK for gfx950, K=1024 and width=262144, including FULL graph replay |

The DP-global gate from #62901, experimental Rust routing policies, the
128-capacity MegaMoE trial, the newer prefill tile/configuration trials, dual
scale cache prototypes and incomplete CPU offload work are excluded.

## Measured balanced decode

DeepSeek-V4-Pro-0813 FP4, 8x MI355X, Rust frontend, TP1/DP8/EP8,
AITER MegaMoE + MoRI, K6 DSpark, synthetic acceptance length 3.77, no offload,
uniform requests per rank, approximately 177K context, two ordered measurements
per framework. The input fingerprint is identical across the listed stages.
The earlier attention checkpoint already includes the fast sparse-attention path;
these numbers do not represent a comparison against unmodified main.

| Stage | c64 cycle ms | c96 cycle ms | c128 cycle ms |
| --- | ---: | ---: | ---: |
| Earlier fast-attention checkpoint | 41.708 | 47.050 | 51.615 |
| Enable FP8 WO_A BMM / preshuffle | 39.821 | 45.109 | 49.301 |
| Router + shared RoPE stack | 39.992 | 45.594 | 49.106 |
| Native E8M0 shared MLP / WO_B | 39.353 | 45.199 | 48.486 |
| Fused query norm + attention/draft E8M0 | 38.196 | 44.353 | 47.547 |
| Selected stack + native indexer TopK | **37.497** | **43.838** | **46.684** |
| Same-host ATOM control | 36.790 | 42.549 | 45.595 |
| Selected latency above ATOM | 1.923% | 3.028% | 2.390% |

Across the measured stages, cycle time falls 10.10%, 6.83% and 9.55% from the
earlier fast-attention checkpoint. WO_A accounts for 4.13-4.52%; E8M0 MLP/WO_B
for 0.87-1.60%; fused query/draft changes for 1.87-2.94%; TopK for 1.16-1.83%,
each relative to its immediate parent. These percentages must not be added.
Router + shared RoPE has operator/memory benefits but no consistent full-cycle
speedup: its measured changes range from a 1.08% regression to a 0.40% reduction.

Each `*_anchored_comparison.json` includes both repetitions and their spread.
For the selected stack the observed range is 0.47%, 0.03%, 0.22% of its median.
These are complete decode cycles including draft, CPU and communication, rather
than isolated GPU target-forward timings. Uniform request counts remove queue
skew but do not force uniform expert loads. Ordered repetitions do not establish
statistical equivalence or remove drift. The ATOM source is pinned to the image
source; the same-host runtime and rebuilt Rust binary differ from the published
nightly. No claim of published ATOM parity follows from this comparison.

## Serving and quality

The selected stack's completed c64 AgentX run used 3600 seconds, ten warmups,
consistent-hash routing, prefix-cache reset and no offload. See
`canonical_c64.json` and `canonical_server.json` for accounting and the exact
recorded launch. There were 6786 successful timed requests, no request errors,
and 15 normal duration-cutoff cancellations.

| c64 metric | Selected stack | Published ATOM target | Relative gap |
| --- | ---: | ---: | ---: |
| Output tok/s/node | 1881.749 | 1984.042 | -5.16% |
| Input tok/s/node | 243301.514 | 254282.443 | -4.32% |
| Total tok/s/node | 245183.263 | 256266.485 | -4.33% |
| P90 interactivity tok/s/user | 59.555 | 65.888 | -9.61% |

The newer paired-query prefill tile reached 1910.556 output tok/s/node but reduced
P90 to 55.058; it is excluded from this PR. Its throughput increase alone does
not establish an overall serving improvement. No final c48/c96/c128 canonical
serving result or genuine-offload c256/c384 result is established.

Real-verification evaluations compare an earlier ASM baseline to the entire
selected stack, not the final TopK dispatch alone. Both arms have 786 scored
responses, unchanged budgets and seed, zero empty outputs and no evaluator
errors. See `quality.json` and `indexer_topk_quality_review_before_canonical.json`.

| Evaluation | Earlier ASM baseline | Selected stack |
| --- | ---: | ---: |
| GPQA | 338/396 | 330/396 |
| AIME | 28/30 | 30/30 |
| NIAH (32K / 128K / 256K, three variants) | 360/360 | 360/360 |

The GPQA difference is -2.02 percentage points, with paired-cluster 95% interval
[-4.80, +0.51]. A meaningful regression remains possible. Sixteen of the 18 lost
outcomes reached the candidate output limit; no filtering or altered budgets
were used. This draft does not claim quality equivalence. NIAH coverage includes
105 actual target query rows on all eight ranks, not just requested concurrency.

## Validation and reproduction

The exact selected source previously passed 11 CPU + 35 real FP8 layer + 71
attention/indexer/compressor + 16 router cases (133 total, no skips).
`validation/summary.json`, JUnit XML and logs contain the fresh runs on this PR
checkout. Commands use the uv-managed environment; AITER and MI355X are required
for GPU cases. The selected public test entry points and environment are recorded
in `validation/summary.json` and `canonical_server.json`.

For example, from this checkout:

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 \
VLLM_ROCM_USE_AITER_FP8BMM=1 VLLM_ROCM_USE_AITER_DSV4_FP8_MXSCALE=1 \
VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA=1 VLLM_MOE_SKIP_PADDING=1 \
.venv/bin/python -m pytest -q \
  tests/kernels/attention/test_rocm_dsv4_mxfp4_indexer.py \
  tests/kernels/moe/test_topk_softplus_sqrt.py::test_dsv4_fast_topk
```

Regression coverage includes changed-input graph replay, actual pooled cache
writers, all valid logits/TopK sets, large byte offsets, short rows, empty batches,
leading dimensions, incompatible-scale fallback and compile/FakeTensor paths.
An additional 17 CPU scheduler/cache-planner cases passed on the draft checkout.
The existing AMD Buildkite kernel steps select the added GPU regressions.
Large-model AgentX and quality runs remain external to CI.

The compound MLP oracle correction is documented separately in
`e8m0_mlp_reference_contract_addendum.json`: independent per-GEMM tolerances were
retained, and the whole quantized pipeline requires exact equality to the same
FP8 operands/scales on the ordinary GEMM path. The initial MLP oracle diagnostic evidence remains in the experiment directory.
`prior_regressions/` includes the original failing and fixed ASM split checks
and the failing delayer scheduling regression.

## Review status

This is a human-review draft. AI assistance was used for implementation, tests,
analysis and PR preparation. The human submitter still needs to review every
changed line, run the relevant validations and defend the change before any
upstream submission. Remaining issues include the GPQA uncertainty, incomplete
canonical concurrency matrix, and upstreaming the integration as appropriately
scoped changes rather than duplicating the adopted proposals.
