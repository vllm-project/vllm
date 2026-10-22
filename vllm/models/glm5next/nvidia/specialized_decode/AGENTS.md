# GLM-5.3-Flash specialized decode (NVIDIA)

A decode executor for small decode and MTP verify steps of GLM-5.3-Flash on
GB300 at TP4, enabled with
`--additional-config '{"decode_backend": "latency_specialized"}'` (the default,
`auto`, keeps the stock decode). With it off, the model runs exactly as without
this package.

## Scope

`specialized_decode_config` (`runtime/__init__.py`) reads the setting once, when
the model is built. `unsupported()` lists every unmet requirement and the model
raises with all of them at startup:

- GPU: SM100 family (GB300 measured), Model Runner V2, no `torch.compile` of the
  model, FULL decode CUDA graphs (`FULL_AND_PIECEWISE` or `FULL_DECODE_ONLY`).
- TP=4 with no pipeline, data, expert, sequence-parallel-MoE or context
  parallelism, no DBO, sleep mode or LoRA.
- The GLM-5.3-Flash geometry in `GEOMETRY`, with mHC. Kernels are compiled for
  these shapes; a new checkpoint with other shapes needs new tactics, not just
  a relaxed check.
- FP8 KV cache; BF16 KDA conv state and FP32 recurrent state; no
  ReplaySSM/RecoverSSM.
- Speculative decoding: none, or MTP with at most 7 tokens.

## Structure

`Glm5NextModelState` (`../model_state.py`) builds a `SpecializedDecode`
(`runtime/decode.py`) once the weights are loaded and binds the KV layouts. On
every forward the backbone asks `plan()` for a `Step`. Only FULL graph captures
of pure decode or uniform verify steps whose token count is a captured size of
at most 64 (and a multiple of `q`) get one. Every other forward, including
profiling, warm-up, PIECEWISE, prefill and mixed steps, runs the stock layers.
The executor reads the stock modules and never modifies them.
`runtime/draft.py` does the same for the MTP layer.

| Module | Stage |
| --- | --- |
| `all_reduce_mhc.py` | Fused Lamport all-reduce + mHC at every sublayer boundary, with the deferred MoE finalize |
| `all_reduce_norm.py` | All-reduce + residual + RMSNorm at the MTP layer's boundaries (no mHC) |
| `kda.py` | Kimi K3's `KdaSplit` for the KDA layers, skinny `in_proj`/`o_proj`, the output gate GEMV |
| `dsa_attention.py`, `indexer.py`, `indexer_query.py`, `kpool.py` | DSA attention around the TRTLLM-Gen sparse MLA, the indexer, the kpool kernels |
| `mega_moe.py`, `grouped_moe.py`, `shared_expert.py` | Per-route expert clusters (1-4 tokens), the expert-grouped persistent kernel (8-64), shared-expert GEMVs |
| `dense_mlp.py` | The three dense layers' MLP |
| `gemm.py`, `linear.py` | BF16 projections on the skinny and split-K GEMMs |
| `logits_gather.py`, `draft_argmax.py` | LM head gather and the vocab-parallel greedy draft argmax |
| `policy.py` | Which implementation each stage runs per token count |

Kernels shared with Kimi K3 (`KdaSplit`, split-K GEMM, `batch_table`) and
DeepSeek-V4.1 (Lamport all-reduce primitives) are imported from those model
directories. Changing them there changes this path: run this package's tests
as well.

## Policy

`policy.DEFAULT` maps each stage to a `batch_table` spec
(`"1-4:per_route,8-64:grouped"`); a token count the spec leaves out runs stock,
and an empty spec turns the stage off. `additional_config["policy"]` replaces
entries by name, which is how to bisect or sweep without code changes:

```bash
--additional-config '{"decode_backend": "latency_specialized",
                      "policy": {"boundary": "", "moe": "1-4:per_route"}}'
```

Each entry's comment records the measured crossover that set it. When a kernel
or the hardware changes, re-measure in situ and update the number with the
spec; do not move a crossover on a microbenchmark alone.

## Invariants

- **Every rank takes the same path.** The boundaries are collectives, so a rank
  that falls back alone hangs the others. `plan()` and every per-step gate read
  only host state all TP ranks share (token count, graph mode, attention
  metadata host fields). Build-time fallbacks depend only on weights and
  config. Never gate on a GPU read-back or a per-rank probe.
- **Sublayer outputs stay TP-partial.** A stage between two boundaries must not
  all-reduce. MoE layers stop before the top-k finalize; the next boundary
  reduces the routed rows and adds the shared expert. A token count without a
  fused boundary must finalize in the MoE kernel (`finalized` in
  `_build_mega`).
- **Mailboxes are ordered by graph replay.** The Lamport buffers rotate through
  `LAMPORT_GENERATIONS` stages, one per boundary, and rely on every launch of the step being in one captured graph.
  Do not move a boundary kernel outside the graph or onto an unjoined stream.
- **Load weights before the PDL wait.** Kernels issue their weight TMA or
  register loads before `griddepcontrol.wait`; activations only after. A new
  kernel that waits first loses most of its gain at 1-8 tokens.
- **Cluster sizes are bounded by GPCs.** GB300 fits 15 eight-CTA clusters with
  one CTA per SM (hence `boundary`'s tokens per cluster) and 7 sixteen-CTA
  clusters (hence two CTAs per SM in the per-route MoE). Check occupancy
  before growing a cluster.
- **Every token count compiles once and all layers share it.** Kernels compile
  at `prepare_kv` for the policy's token counts; keep per-layer state in
  tensors, not in compile keys.
- The NVLink logits gather and split-K head run only outside graph capture
  (MRV2 computes logits eagerly); captured steps use NCCL.

## Kernels

- CuTe DSL for performance-sensitive kernels; no new Triton (MRV2-style
  prepare kernels excepted).
- Numerics: a kernel's error against an FP64 reference must be no larger than
  the stock kernel's. Where stock is a quantizer or a cache write (Hadamard FP8
  query, kpool, KV cache bytes), match it bit for bit.
- Tests live under `tests/models/glm5next/test_specialized_decode_*.py`, compare
  against FP64 or naive PyTorch references, and assert the error is no larger
  than the stock kernel's. 4-GPU tests skip with fewer GPUs; CI has no GB300, so
  run them locally.

## Validation

- Step time: nsys CUDA graph replay span of one step, minimum over the TP
  ranks. Run-to-run noise is about ±0.25 ms at 32 tokens; re-measure the
  baseline in the same session and repeat before claiming a win.
- Served ITL: `vllm bench serve` without prefix caching and with
  `--numa-bind` on both builds. Unbound, a worker on the other Grace socket
  adds about 2.4 ms per step at concurrency 1, which the stock path hides.
- Greedy A/B: serial runs first. For batched A/B run the engine core in
  process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); with a multiprocess core the
  prompts reach the scheduler one by one and the prefill split varies per
  launch, so stock's own outputs change. Divergences should sit at near-ties
  (top-1/top-2 logprob margin around 0.25 or less); bisect a larger one with
  `policy` overrides.
- Output-affecting changes: GSM8K with and without MTP; earlier builds score
  0.910-0.925 against stock's 0.912-0.914.
