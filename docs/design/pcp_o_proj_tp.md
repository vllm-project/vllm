# PCP O-Proj TP MVP Design

Implementation baseline: `vllm-project/vllm upstream/main@f8e0602713`.

## Background

The fine-grained O-Proj tensor parallelism in vllm-ascend further shards the
O-Proj input dimension across additional ranks beyond TP, reducing the resident
weight on each rank. In the DSA-CP implementation, the runtime path depends on
the batch type: decode uses the finer-grained weight shards and reduces their
partial sums, while a batch containing prefill first gathers the original
TP-sized weight shard and then runs the standard O-Proj computation.

vLLM already has independent TP, PCP, and DCP process groups, but the generic
`RowParallelLinear` shards weights only across TP ranks. This MVP uses PCP as a
second O-Proj weight-sharding axis and provides an equivalent dynamic path under
the name `pcp_o_proj_tp`.

## MVP limitations

- `prefill_context_parallel_size > 1`.
- `decode_context_parallel_size = 1`.
- Only DeepSeek V2/V3/V3.2 and GLM-4 MoE Lite MLA are supported. GLM-4 MoE Lite
  attention inherits the DeepSeekV2 MLA path, so it reuses the same explicit
  prefetch and dynamic O-Proj implementation.
- Only unquantized FP16/BF16 weights with `bias=False` are supported.
- Only eager execution is supported. CUDA Graph, `torch.compile`, and DBO are
  not supported yet.
- LoRA is not supported yet.
- CPU weight offloading is not supported yet.
- The attention module must explicitly call
  `o_proj.prefetch_full_weight_if_needed(has_prefill)` before the main attention
  computation.

Configuration validation and linear-layer construction checks enforce all of
these restrictions and fail fast instead of silently ignoring the feature.

## Weight layout

Let the O-Proj weight be `W[N, K]`, the TP size be `T`, and the PCP size be `P`.
Standard row parallelism keeps the following shard on TP rank `t`:

```text
W_tp[t] = W[:, t*K/T : (t+1)*K/T]
```

The MVP further shards each TP shard across PCP rank `p`:

```text
W_local[t,p] = W[:, (t*P+p)*K/(T*P) : (t*P+p+1)*K/(T*P)]
```

The checkpoint loader uses flattened rank `t*P+p` and world size `T*P`, so no
checkpoint format change is required. The resident O-Proj weight on each rank
is reduced to `1/P` of the standard TP layout.

## Runtime paths

### Decode-only batches

PCP rank `p` takes the corresponding feature slice from the TP-sharded
attention output `X_t[..., K/T]` and multiplies it by the local
`W_local[t,p]`. The partial sums are first all-reduced over the PCP group to
recover the standard O-Proj partial result for a TP rank. The existing TP
reduction semantics are then preserved.

```text
Y_t = PCP-AllReduce(X_t,p @ W_local[t,p]^T)
Y   = TP-AllReduce(Y_t)                    # when reduce_results=True
```

DeepSeek-V3.2 O-Proj uses `reduce_results=False`, so this path performs only the
PCP reduction. The existing fused TP all-reduce/RMSNorm operation remains
responsible for the TP reduction.

### Prefill or mixed batches

The attention module determines the batch type from the layer's
`MLACommonMetadata.num_prefills`. If the batch contains any prefill requests,
it asynchronously all-gathers the weight over PCP before the main attention
computation:

```text
W_local[t,p]^T --PCP AllGather--> W_tp[t]^T
```

O-Proj waits for the asynchronous handle only when its forward method is
reached, then applies `W_tp[t]` to the local tokens on the current PCP rank.
This path does not perform a PCP output reduction because different PCP ranks
process different tokens. The existing TP reduction remains unchanged.

To concatenate along dimension 0 with `all_gather_into_tensor`, the collective
input is a contiguous transposed weight with shape `[K/(T*P), N]`, and the
output buffer has shape `[K/T, N]`. The full TP weight is retained only as a
transposed view, avoiding an additional copy.

## Buffer lifecycle

The full TP weight buffer is shared by vLLM config scope, device, dtype, and
shape. It is allocated during model construction, and all compatible layers
hold the same shared buffer, so the cost is included in startup memory
budgeting. With normal sequential layer execution, one layer waits for and
consumes its gathered weight before the next layer reuses the buffer. The
resident overhead is therefore one TP-sized O-Proj shard rather than one copy
per layer.

The input to the asynchronous collective is a contiguous transpose of the
current layer's local weight. It remains alive until the handle has completed
and is then released. DBO is currently disabled to prevent two concurrent
microbatches from using the shared buffer at the same time.

## Explicit triggering and correctness constraints

Both the generic MLA wrapper and the modular DeepSeek-V3.2 attention path
explicitly trigger the prefetch. `PCPOProjRowParallelLinear.forward()` requires
the trigger to have occurred before every invocation. A missing trigger raises
an error instead of inferring the batch type. This guarantees that:

- the prefill/mixed gather starts early enough to overlap with Q/K/V
  projections, RoPE, the indexer, and the main attention computation;
- decode-only batches do not issue an unnecessary weight gather;
- O-Proj does not read the shared buffer before the asynchronous gather has
  completed; and
- profiling paths without metadata use the decode-sharded calculation, which
  preserves correctness for zero attention outputs.

## Configuration

The MVP adds the following CLI/configuration option:

```text
--enable-pcp-o-proj-tp
```

Example:

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 1 \
  --enable-pcp-o-proj-tp \
  --enforce-eager \
  --dtype bfloat16
```

## MVP validation results

Online accuracy was evaluated on August 21, 2026, using four H100 GPUs,
GLM-4.7-Flash BF16, and the 1,319-question GSM8K test set. TP4, PCP4TP1, and
PCP2TP2 used eager execution, `max_model_len=8192`, 5-shot prompting,
`temperature=0`, thinking disabled, and concurrency 32. The 32-question gate
used `max_tokens=256`; the full evaluation used `max_tokens=4096`.

| Parallel strategy | 32-question gate | Full-set correct | Full-set accuracy | Failed requests | Invalid answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 baseline | 25/32 | 1134/1319 | 85.9742% | 0 | 0 |
| PCP4TP1 | 25/32 | 1136/1319 | 86.1259% | 0 | 0 |
| PCP2TP2 | 26/32 | 1135/1319 | 86.0500% | 0 | 0 |

Relative to the TP4 baseline, PCP4TP1 changed 35 answers from incorrect to
correct and 33 from correct to incorrect, for a net gain of two. PCP2TP2
changed 32 answers from incorrect to correct and 31 from correct to incorrect,
for a net gain of one. The three reduction topologies introduce numerical
differences large enough to alter some autoregressive generation paths, so
per-question text is not expected to match exactly. No accuracy regression was
observed over the full dataset.

Based on checkpoint safetensors metadata, the O-Proj weights across 48 layers
total `1,006,632,960` bytes, or `3.2242%` of all model parameter bytes. With the
feature enabled:

- PCP4TP1 reduces the resident O-Proj weight on each rank from 100% of the full
  O-Proj weight to 25%, a 75% reduction; and
- PCP2TP2 reduces the resident O-Proj weight on each rank from 100% of the
  original TP2 shard to 50%, a 50% reduction.

These percentages describe parameter weights only. They exclude the CUDA
context, KV cache, communication buffers, the shared single-layer prefetch
buffer, and allocator fragmentation, and therefore do not represent the total
HBM reduction.

## Future work

1. Add per-layer tensor comparisons for TP-by-PCP prefill, mixed, and decode
   paths, multi-run stability tests, and traces of weight-gather/attention
   overlap.
2. Integrate the shared-buffer lease with a concurrency-safe model-level
   workspace manager and remove the DBO restriction.
3. Add CUDA Graph and `torch.compile`-capturable paths.
4. Define joint gathering of weights, scales, and metadata for FP8, block
   quantization, and other quantized formats.
5. Evaluate a dedicated PCP communicator or stream to avoid serialization with
   attention PCP collectives.
