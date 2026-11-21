# Hierarchical LM head (experimental)

At low concurrency, repeatedly projecting draft states onto a large vocabulary
can contribute noticeably to speculative-decoding latency. A trained hierarchical
LM head first selects vocabulary clusters, then scores only their token rows.
This reduces draft-head work while retaining the original BF16 vocabulary weights.
Selection is approximate and may reduce draft acceptance.

The shared `HierarchicalLMHead` layer supports Gemma4 MTP and NemotronH MTP.
The hierarchical head compiles
`(lm_head_weights[sel] * hidden[:, None, :]).sum(-1)`, keeping the
gather inside compilation so Inductor can fuse it with the reduction. Token
count is dynamic; model dimensions are static. It reads the original vocabulary
weights directly, with no reordered full-head copy or additional CUDA sources.

`get_top_tokens` returns selected vocabulary IDs without allocating full logits.
`forward` returns full-vocabulary logits with unselected tokens masked. Numerical
rounding can differ from a dense matrix multiplication. Actual lowering and
performance depend on PyTorch and hardware.

## Checkpoint configuration

For an embedded router, enable the head in the **draft checkpoint's** `config.json`:

```json
{
  "use_ordered_embeddings": true,
  "num_centroids": 4096,
  "centroid_intermediate_top_k": 32
}
```

These fields supplement the model's existing config; they do not replace it.
An embedded router includes these tensors, loaded through the normal
checkpoint loader (alternatively, supply them in a sidecar as described below):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `masked_embedding.centroids.weight` | `[num_centroids, hidden_size]` | Trained cluster signatures, floating point |
| `masked_embedding.token_ordering` | `[vocab_size]` | INT64 permutation of vocabulary IDs, grouped by cluster |

Clusters must be balanced: `vocab_size` must be divisible by `num_centroids`.
Each consecutive group of `vocab_size / num_centroids` ordering entries defines
one cluster. Missing router tensors, incompatible shapes, and invalid
permutations are rejected during loading. With `use_ordered_embeddings` absent
or false and no sidecar supplied, the dense path remains enabled.

NemotronH currently requires TP=1 and unquantized, shared vocabulary weights.
It can load a sidecar alongside the original bundled MTP checkpoint, or load
these router tensors from an existing hierarchical draft checkpoint. The shared
LM head stays unchanged.
Gemma4 retains its existing embedding-sharing and tensor-parallel weight-gathering
behavior.

## Runtime router and top-K options

Set these optional fields in `--speculative-config`:

| Field | Behavior |
| --- | --- |
| `hierarchical_lm_head_router_path` | Local `.pt` or `.safetensors` sidecar containing the two `masked_embedding.*` tensors above. Overrides embedded router tensors. |
| `hierarchical_lm_head_top_k` | Number of clusters selected per draft token. Required with a sidecar; otherwise overrides `centroid_intermediate_top_k` from the draft config. |

A sidecar enables the hierarchical head without changing or repackaging the
original model or MTP checkpoint. The number of clusters is inferred from its
centroid tensor. PyTorch sidecars are loaded on CPU with `weights_only=True`;
additional training metadata is ignored. The same hidden-size, finite-weight,
balanced-cluster and vocabulary-permutation checks apply to both formats.
Top-K must be between one and the number of clusters. A top-K override without
an embedded or sidecar router is rejected. Configuration validation is model-agnostic. Serving currently wires the shared
head into NemotronH and Gemma4 MTP; other drafters need a model integration. Local sidecars must be readable by every worker.

### NemotronH: original bundled MTP plus a sidecar

[jhuang/n3.5-lightning-HLM](https://huggingface.co/jhuang/n3.5-lightning-HLM)
provides a community router example for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. It is not an official NVIDIA
hierarchical-head checkpoint. Supply the exported router directly:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7,"use_local_argmax_reduction":true,"hierarchical_lm_head_router_path":"/path/to/router.pt","hierarchical_lm_head_top_k":32}'
```

The backbone, shared LM head and MTP weights load from the original model;
only the router loads from the sidecar. With no sidecar and no embedded router,
NemotronH retains its dense LM head. No model-specific environment variables or
separate draft model path are required.

### Gemma4: existing assistant checkpoint

The upstream launch command is unchanged. The official assistant contains its
router and default `centroid_intermediate_top_k`:

```bash
vllm serve google/gemma-4-E4B-it \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":7,"use_local_argmax_reduction":true}'
```

To override only top-K, add `"hierarchical_lm_head_top_k":16` to that speculative
config. The checkpoint's config and router weights remain unchanged. As with
NemotronH, a sidecar can explicitly replace the embedded router, in which case
an explicit top-K is required.

A router is specific to its vocabulary, LM-head weights and draft-state
distribution; use a router trained and evaluated for the selected checkpoint.

## Performance and validation

In the November 20, 2026 single-GB200 experiment with Nemotron-3.5-Lightning
BF16, TP=1, 7 speculative tokens, C4096/K32 and the full SpecDec-Bench
qualitative split (880 conversations), compiled hierarchical scoring improved
end-to-end OTPS from 649.32 to 715.96 tokens/s (**+10.26%**) at concurrency 1.
Mean acceptance length decreased from 3.8997 to 3.8502 (**-1.27%**).
These measurements precede the subsequent router configuration and backend API
cleanup. Each method was measured in one full-split run; run-to-run variability
was not measured.

### Gemma4 MTP comparison with upstream

A paired comparison against upstream commit `8586ccbcf58e69ecb69e370dd2fa69fb739f9b0c`
(the base of this branch) used the official `google/gemma-4-E4B-it` target and
`google/gemma-4-E4B-it-assistant`, with the original C2048/K32 router configuration.
Both implementations ran on a single GB200 with BF16, TP=1, seven speculative
tokens, temperature zero, an 8192-token output cap and the full qualitative split
(880 conversations / 1138 turns). Model and dataset revisions were pinned.

| Concurrency | Upstream AL | HLM AL | AL change | Upstream OTPS | HLM OTPS | OTPS gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.3513 | 3.3507 | -0.02% | 559.42 | 562.31 | +0.52% |
| 32 | 3.3388 | 3.3468 | +0.24% | 10653.02 | 11386.19 | +6.88% |

The compiled implementation preserves mean acceptance length closely while
improving observed OTPS, especially at concurrency 32. Native acceptance counters
confirm the same pattern (AL changes -0.014% and +0.268%, respectively).
Each cell is one full-split run; the small concurrency-1 throughput difference
has not been established through repeated measurements.

This is an acceptance-length comparison, not a bitwise generation-equivalence
claim. Exact decoded response text matched on 1136/1138 turns at concurrency 1
and 333/1138 at concurrency 32. Numerical execution and batch scheduling can
vary between implementations.

An isolated head benchmark with the actual assistant weights, CUDA graphs,
20 rotating hidden-state inputs and three timing repeats measured speedups over
the upstream head of 1.12x, 1.51x and 2.30x at batches 1, 8 and 32. Selected
vocabulary IDs matched the reference exactly, logits passed the common FP32
reference checks, and top-1 IDs agreed on all tested inputs. These component
speedups are separate from the end-to-end OTPS gains above.

For a layer benchmark, compile separately from steady-state CUDA graph timing:

```bash
python benchmarks/kernels/benchmark_hierarchical_LM_head.py \
  --num-centroids 4096 --top-k 32 --batch-sizes 1 2 4 8 16 32
```

Random weights measure execution time, not routing quality. Use a compatible
trained checkpoint and a real speculative-decoding workload for acceptance
length and end-to-end throughput measurements.

For a Gemma4 layer comparison, point the benchmark at a local official
assistant snapshot. It uses its actual BF16 head and router weights:

```bash
python benchmarks/kernels/benchmark_hierarchical_LM_head.py \
  --checkpoint /path/to/gemma-4-E4B-it-assistant \
  --batch-sizes 1 2 4 8 16 32 --repeats 6 \
  --input-pool-sizes 1 20 --report-stages
```

The benchmark compares compiled hierarchical scoring against dense BF16.
Optionally add `--upstream-source /path/to/upstream/vllm/model_executor/models/gemma4_mtp.py`
to include upstream's masked embedder as an external reference. Compilation is
excluded from CUDA graph timing.

`--input-pool-sizes 1 20` reports fixed-input replay and an explicitly unrolled
20-input cycle. All methods see the same hidden-state pool. `--report-stages`
adds routing, scoring with precomputed routes, and sparse argmax timings. These
isolated timings are diagnostics and are not additive because fusion and cache
behavior differ from the full head.
