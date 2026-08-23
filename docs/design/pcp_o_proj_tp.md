# PCP O-Proj TP MVP Design

Implementation baseline: `vllm-project/vllm upstream/main@b26039b09f`.

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
- O-Proj itself must use the unquantized linear method. Quantized checkpoints
  remain usable when their quantization configuration excludes O-Proj, as the
  PCP-validated GLM-5.2-NVFP4 checkpoint excludes all self-attention modules.
  A quantized O-Proj fails during post-load processing.
- `bias=False` is required.
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

## Weight-switch scope

The implementation follows the vllm-ascend `TPWeightSwitchMixin` mechanism but
opts in only `UnquantizedLinearMethod`. The method declares its `weight` tensor
and post-load input-shard dimension; the mixin owns the reusable buffers,
asynchronous collective handle, and local/full tensor aliases.

This boundary matches the quantized model coverage already present for PCP.
The upstream PCP GSM8K matrix contains GLM-5.2-NVFP4, whose ModelOpt
configuration excludes every `self_attn` module from NVFP4 quantization. Its
O-Proj therefore uses the same unquantized method as a BF16 checkpoint. The
supported method set is intentionally limited to `UnquantizedLinearMethod`.

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

The model runner records whether the logical, pre-partition batch contains any
prefill requests before PCP splits its tokens. Every PCP rank therefore makes
the same collective decision, including when prefix caching leaves a rank with
decode tokens while another rank receives a short prefill. If the logical batch
contains prefill, the attention module asynchronously all-gathers the weight
over PCP before the main attention computation:

```text
W_local[t,p]^T --PCP AllGather--> W_tp[t]^T
```

O-Proj waits for the asynchronous handle only when its forward method is
reached, then applies `W_tp[t]` to the local tokens on the current PCP rank.
This path does not perform a PCP output reduction because different PCP ranks
process different tokens. The existing TP reduction remains unchanged.

Because the weight-sharded axis is not dimension 0, the mixin moves that axis
to the front and makes a contiguous collective input for
`all_gather_into_tensor`. If the target kernel accepts the resulting moved-dim
view, the collective output is used directly. Kernels that require a contiguous
post-load layout request a shared assembly buffer and copy the gathered view
into it after the asynchronous handle completes.

## Buffer lifecycle

Full TP buffers are shared by vLLM config scope, method, tensor attribute,
device, dtype, and shape and are allocated during model construction. All
compatible layers hold the same shared full-weight buffer, so its cost is
included before profiling and KV-cache sizing. Each layer also retains a local
contiguous staging tensor for the dimension-1 collective input. With normal
sequential layer execution, one layer waits for and consumes the shared gathered
weight before the next layer reuses the full-weight buffer.

The input to each asynchronous collective is either the local tensor itself or
a persistent contiguous moved-dim staging tensor. Non-leading staging inputs
are refreshed before every gather so in-place weight updates are visible. DBO
is currently disabled to prevent two concurrent microbatches from using the
shared buffers at the same time.

## Explicit triggering and correctness constraints

Both the generic MLA wrapper and the modular DeepSeek-V3.2 attention path
explicitly trigger the prefetch after their Q/KV normalization and before the
remaining query projection and main attention computation. The trigger uses
the logical pre-partition batch type propagated through `ForwardContext`, so
all PCP ranks preserve collective ordering.
`PCPOProjRowParallelLinear.forward()` requires the trigger to have occurred
before every invocation. A missing trigger raises an error instead of inferring
the batch type. This guarantees that:

- the prefill/mixed gather avoids contending with Q/KV RMSNorm while retaining
  the remaining query projection, RoPE, the indexer, and the main attention
  computation as overlap candidates;
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

The following end-to-end results cover the unquantized O-Proj path. The unit
tests cover asynchronous gather, wait-before-use, full/local alias switching,
restoration after a failed linear call, logical-versus-local prefill decisions,
and rejection of a quantized O-Proj method.

Online accuracy was evaluated on August 23, 2026, using four NVIDIA H20 GPUs,
GLM-4.7-Flash BF16, and the 1,319-question GSM8K test set. TP4, PCP4TP1, and
PCP2TP2 used eager execution, DCP1, 5-shot prompting, `temperature=0`, seed 42,
concurrency 32, and `max_tokens=4096`. Each topology first passed a 32-question
output gate, then completed the full dataset.

| Parallel strategy | 32-question gate | Full-set correct | Full-set accuracy | Failed requests | Invalid answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 baseline | 25/32 | 1129/1319 | 85.5951% | 0 | 0 |
| PCP4TP1 | 25/32 | 1135/1319 | 86.0500% | 0 | 0 |
| PCP2TP2 | 24/32 | 1131/1319 | 85.7468% | 0 | 0 |

PCP4TP1 and PCP2TP2 differ from TP4 by +0.4549 and +0.1516 percentage points,
respectively. The three reduction topologies can follow different
autoregressive generation paths because their floating-point reduction orders
differ.

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

The same four-H20 host was used to measure process-level memory with a fixed
32 GiB KV cache. `Model loading took` is the model runner's allocated-memory
measurement after loading and post-load processing. Ready HBM is the
`nvidia-smi` value after engine initialization.

| Parallel strategy | Feature | Model-load memory/rank | Ready HBM/rank |
| --- | --- | ---: | ---: |
| PCP4TP1 | disabled | 17.02 GiB | 59,461 MiB |
| PCP4TP1 | enabled | 16.58 GiB | 60,021 MiB |
| PCP2TP2 | disabled | 15.15 GiB | 56,541 MiB |
| PCP2TP2 | enabled | 15.17 GiB | 57,140 MiB |

PCP4TP1 reduces model-load allocation by 0.44 GiB per rank. For PCP2TP2, the
smaller local parameter shard and the current persistent gather staging have
similar sizes, yielding a 0.02 GiB measured increase. Ready HBM includes the
communication and runtime workspaces initialized by the feature and is 560 MiB
and 599 MiB higher for PCP4TP1 and PCP2TP2, respectively. These measurements
distinguish the parameter-layout reduction from total serving-process HBM.

Serving performance used random 32,768-token inputs and 1,024 generated tokens,
`ignore_eos`, `temperature=0`, seed 12345, prefix caching disabled, and a fixed
32 GiB KV cache. Each row is one completed controlled run; all requests
succeeded. Throughput is generated-token throughput.

| Topology | Feature | Concurrency | Mean TTFT | Mean TPOT | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| PCP4TP1 | disabled | 1 | 2,027.49 ms | 82.33 ms | 11.87 tok/s |
| PCP4TP1 | enabled | 1 | 2,030.59 ms | 86.39 ms | 11.33 tok/s |
| PCP4TP1 | disabled | 8 | 9,368.98 ms | 89.98 ms | 80.51 tok/s |
| PCP4TP1 | enabled | 8 | 9,402.45 ms | 94.69 ms | 76.84 tok/s |
| PCP4TP1 | disabled | 16 | 17,376.00 ms | 97.19 ms | 139.50 tok/s |
| PCP4TP1 | enabled | 16 | 17,438.39 ms | 104.49 ms | 131.04 tok/s |
| PCP2TP2 | disabled | 1 | 2,091.03 ms | 89.73 ms | 10.91 tok/s |
| PCP2TP2 | enabled | 1 | 2,151.26 ms | 95.16 ms | 10.29 tok/s |
| PCP2TP2 | disabled | 8 | 9,008.10 ms | 95.98 ms | 76.17 tok/s |
| PCP2TP2 | enabled | 8 | 8,968.01 ms | 103.23 ms | 71.27 tok/s |
| PCP2TP2 | disabled | 16 | 16,404.51 ms | 103.03 ms | 133.75 tok/s |
| PCP2TP2 | enabled | 16 | 16,417.18 ms | 108.64 ms | 127.70 tok/s |

Across concurrency 1/8/16, enabling the feature changes mean TTFT by
`+0.15%/+0.36%/+0.36%` for PCP4TP1 and `+2.88%/-0.45%/+0.08%` for PCP2TP2.
Mean TPOT changes by `+4.93%/+5.23%/+7.51%` and
`+6.05%/+7.55%/+5.45%`, respectively; generated-token throughput changes by
`-4.55%/-4.56%/-6.06%` and `-5.68%/-6.43%/-4.52%`.

## PCP8 parameter projections

The following projections use Hugging Face checkpoint tensor metadata and
assume TP1, PCP8, DCP1, and an implementation compatible with each model's
O-Proj representation. They include the current single-layer full-weight
prefetch buffer and exclude communication, staging, and allocator overhead.

| Model/checkpoint | Total parameter bytes | Shardable O-Proj | Local O-Proj at PCP8 | Per-rank persistent saving | Resulting persistent weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.2 BF16 | 1,403.186 GiB | 14.812 GiB | 1.852 GiB | 12.773 GiB | 1,390.413 GiB |
| GLM-5.2 FP8 | 703.723 GiB | 7.408 GiB | 0.926 GiB | 6.388 GiB | 697.335 GiB |
| DeepSeek-V4 Flash-Base | 274.436 GiB | 1.375 GiB | 0.172 GiB | 1.172 GiB | 273.264 GiB |
| DeepSeek-V4 Flash | 148.648 GiB | 1.375 GiB | 0.172 GiB | 1.172 GiB | 147.476 GiB |
| DeepSeek-V4 Pro-Base | 1,495.734 GiB | 6.782 GiB | 0.848 GiB | 5.825 GiB | 1,489.910 GiB |
| DeepSeek-V4 Pro | 805.319 GiB | 6.782 GiB | 0.848 GiB | 5.825 GiB | 799.495 GiB |

For a combined TP size `T`, the local O-Proj, prefetch buffer, and per-rank
saving in this table are divided by `T`.

## Future work

1. Add per-layer tensor comparisons for TP-by-PCP prefill, mixed, and decode
   paths, multi-run stability tests, and traces of weight-gather/attention
   overlap.
2. Integrate the shared-buffer lease with a concurrency-safe model-level
   workspace manager and remove the DBO restriction.
3. Add CUDA Graph and `torch.compile`-capturable paths.
4. Extend the O-Proj method set together with representation-specific
   reconstruction and an end-to-end PCP model validation.
5. Evaluate a dedicated PCP communicator or stream to avoid serialization with
   attention PCP collectives.
