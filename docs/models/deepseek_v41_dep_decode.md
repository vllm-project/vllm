# DeepSeek-V4.1 specialized decode with data and expert parallelism

The experimental DEP executor keeps attention at TP1. Each DP rank owns its
request rows, while experts are sharded across 2, 4 or 8 GPUs on one NVLink
node. Local routed experts retain the original TRTLLM W13/W2 kernels.

## Enable

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --additional-config '{"decode_backend":"latency_specialized_dep"}' \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

Other model limits, tokenizer and memory settings remain deployment choices.
Set `decode_backend` back to `auto` to disable the executor.

For a different group size, change `--data-parallel-size`. The same
`latency_specialized_dep` backend uses the configured DP/EP group size. Keep
`--tensor-parallel-size 1` and `--enable-expert-parallel`; these are not the
TP4/TP8 executors.

| GPUs | `--data-parallel-size` | Routed experts per rank (384 total) | Maximum decode rows per rank |
| ---: | ---: | ---: | ---: |
| 2 | 2 | 192 | 128 |
| 4 | 4 | 96 | 128 |
| 8 | 8 | 48 | 384 |

Unsupported layouts fail at startup. Global expert IDs are preserved through
dispatch.

## Supported Scope

- SM103 GPUs, TP1 with equal DP and EP sizes as above, CP1/PP1, and modular TRTLLM MoE with linear
  expert placement. No sequence parallelism, EPLB or microbatching.
- DSpark5 with adaptive verification disabled. Sampling is unchanged; the
  TP-sharded Markov sampler is not used across independent DP requests.
- FULL decode graphs, FP8 KV, and the sparse MXFP4 indexer shown above.
- KV connectors compatible with the model's cache layout and FULL CUDA graphs.
  Connectors returning `True` from `requires_piecewise_for_cudagraph()` are rejected,
  including a `MultiConnector` with any child that requires PIECEWISE graphs.
- No LoRA, sleep, weight updates or batch-invariant mode.

Unsupported configurations fail at initialization. Pure decode graphs up to
128 token rows per rank for DEP2/4, or 384 for DEP8, use the executor. Prefill,
eager execution and larger graphs retain the original layer loop. Request
counts and token rows differ: DSpark5 can produce six target rows and five
draft rows per request. The limit applies to the padded graph size, not just
the number of live tokens.

The executor reuses the specialized attention/indexer and Engram prefetch
paths, adds local mHC boundaries without TP collectives, and fuses top-k with
activation quantization before expert dispatch.

For uniform, nonempty decode graphs, expert contributions are published
directly to the rank that owns each request. The consumer fuses the reduction,
local shared-expert addition, mHC, normalization and activation quantization.
Eligible target layers also write routing and quantization outputs directly
into a symmetric dispatch buffer, and fuse expert finalize with publication.
The target and draft share three rotating owner buffers on the model stream.
Accumulation is in FP32 in rank order before rounding to BF16; it is not
bitwise identical to NCCL's reduction tree.

Before Engram and at the final layer, the original EP combine is retained.
Direct dispatch additionally requires an even row count. Empty or uneven
rank layouts use the standard modular all-gather/reduce-scatter path, with
every rank participating. No request-local gate skips a required collective.

Performance depends on the per-rank row count and context distribution. This
does not change the default backend. Two-rank timings do not predict a four-
or eight-rank speedup: compare against `auto` with the same DP/EP width.
In benchmark tables, C2/C4/C8 denote requests **per DP rank**, not total
concurrency. Total concurrency is the per-rank count multiplied by the DP size.

## Disaggregated prefill and decode

Keep the prefill instance on `decode_backend=auto`. On the decode instance,
use `latency_specialized_dep` with the connector's normal PD configuration.
For example, the validated NIXL pull setup adds:

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
```

On prefill, use the same connector with `kv_role=kv_producer`. The default
transfer mode is pull; `NixlPullConnector` is also accepted. Both instances
must use compatible model and KV-cache settings. TP remains 1, but their DP/EP
widths need not match: a DEP2 prefill can feed a DEP2 or DEP4 decode instance.
Each instance's EP group stays local to its own NVLink-connected GPUs.

For this pull setup, forward the prefill response's `kv_transfer_params` to
decode using the [NIXL disaggregated-serving flow](../features/nixl_connector_usage.md).
The scheduler waits for remote KV completion before admitting the request.
Initial prompt replay and mixed prefill/decode batches keep the stock path.
Eligible subsequent FULL decode graphs use the DEP executor.

The executor has no connector-name, transfer-mode or buffer-device whitelist.
It resolves the configured class through `KVConnectorFactory` and checks
`requires_piecewise_for_cudagraph()` with the connector's extra configuration.
This also applies to dynamically loaded connectors and `MultiConnector`.
Modes requiring layer-wise Python callbacks cannot run inside FULL replay.
Normal connector checks for model support, hybrid KV caches, layout and
transfer settings still apply; passing the graph check does not override them.

KV registration, load/save synchronization, transfer completion and block
lifetime remain with the existing runner and connector. The executor reads
and writes the original registered caches. Other connectors use their own
serving flow, such as [NIXL push](../design/nixl_kv_push_connector.md), without
DEP-specific transport code. End-to-end PD validation here covers NIXL pull;
the configuration tests cover the generic admission check, not transfers for
every connector.

## DEP8 Larger Decode Batches

DEP8 automatically uses a 384-row capacity with the same
`latency_specialized_dep` backend. No extra capacity flag is needed. With
DSpark5, 32 and 64 requests per rank produce 192 and 384 target rows before
graph padding. Capture the corresponding target and draft graph sizes to use
this path. DEP2/4 remain limited to 128 rows. Dispatch copies every activation
row; there is no route-aware activation filtering. Expert GEMMs, quantization
and sampling remain unchanged.

### Paired Measurements

Eight B300 GPUs, TP1 and DP=EP=8; the same 84-648 Ki-token input distribution,
3,000 output tokens per request, DSpark5, AV off, and synthetic acceptance
length 3.51 on both backends. Values are medians of three rounds. Each count
starts with a prefix-cache reset, serial prefill and a 256-token warmup;
timed input-token cache hits were about 99.98%, with no preemption. Baseline
and candidate ran sequentially, not interleaved.

| Requests per rank (total) | Target rows per rank | Main step (ms) | DEP step (ms) | Time saved | Main output (tokens/s) | DEP output (tokens/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (64) | 48 | 14.620 | 12.472 | 14.69% | 15,371 | 18,012 |
| 21 (168) | 126 | 20.951 | 19.087 | 8.89% | 28,105 | 30,851 |
| 32 (256) | 192 | 23.452 | 21.589 | 7.95% | 38,339 | 41,653 |
| 64 (512) | 384 | 32.047 | 30.155 | 5.90% | 56,173 | 59,684 |

Step time uses the original benchmark's host-inclusive common streaming
window at full concurrency, divided by the mean number of output chunks per
request. It includes target and draft work, but is not a CUDA-event kernel
timer or prefill-inclusive E2E time. Output throughput is the aggregate across
all eight GPUs in that window. FULL graph replay and the specialized path
were recorded on every rank, including the 192- and 384-row captures.

### GSM8K

Separate accuracy runs used all 1,319 five-shot GSM8K questions, real block
rejection, seed 42, temperature 1.0, top-p 0.95, thinking enabled and no
request-level output cap. Invalid final-answer formats count as incorrect.

| Requests per rank | Main correct | DEP correct |
| ---: | ---: | ---: |
| 32 | 1,238 / 1,319 (93.86%) | 1,238 / 1,319 (93.86%) |
| 64 | 1,246 / 1,319 (94.47%) | 1,257 / 1,319 (95.30%) |

All four runs completed without request errors or length truncation. These
single-seed scores are regression evidence, not a claim of improved accuracy
or bitwise equivalence.

## Experimental C8 Attention

`dep_wide_c8_attention: true` in `additional_config` enables an experimental
64-head sparse attention kernel for the target's 48-row, 640-column decode
case. It is off by default and allocates about 30 MiB of fixed scratch per
rank. Other shapes, the draft, and unsupported layouts retain their existing
dispatch. Component correctness has been exercised through 384 rows, but the
larger shapes regressed and are not selected. The DEP8 capacity extension
does not select this attention kernel for those larger shapes. Full-model
validation of the optional attention kernel is still required before deployment.
