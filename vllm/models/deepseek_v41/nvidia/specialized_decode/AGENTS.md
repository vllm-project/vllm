# DeepSeek-V4.1 specialized decode (NVIDIA)

A decode executor for small decode batches of DeepSeek-V4.1 on GB300, enabled
with `--additional-config '{"decode_backend": "latency_specialized"}'` (the
default, `auto`, keeps the stock decode). With it off, the model runs exactly as
without this package. The executor runs the target model; the DSpark draft
model keeps its stock paths.

## Scope

`specialized_decode_config` (`__init__.py`) reads the setting once, when the
target model is built. Any deployment other than the one below raises at
startup on every TP rank with the missing requirements:

- GPU: GB300 (SM103).
- Runner: Model Runner V2.
- Parallelism: TP=4; no pipeline, data, expert or context parallelism, no
  micro-batching.
- Speculative decoding: DSpark, without adaptive verification.
- Attention: `FLASHINFER_MLA_SPARSE_DSV41`, fp8 KV cache, MXFP4 indexer with
  sparse logits.
- CUDA graphs: FULL graphs for decode batches only (`FULL_AND_PIECEWISE` or
  `FULL_DECODE_ONLY`).
- No `torch.compile` of the model, no sleep mode, no runtime weight updates.
- No LoRA, KV transfer or batch invariance.
- DeepGEMM and NVLink multicast for the fused all-reduce + mHC.

This check runs once, at startup. The configuration is the same on every
rank, but the last requirement is probed per rank: the custom all-reduce sets
up its NVLink multicast buffers on each rank and catches a failure there, so
one rank can be without them while the others have them. The ranks' results
are therefore combined with a MIN all-reduce over the TP group: if any rank
misses a requirement, every rank raises, rather than some ranks running the
specialized decode and others the stock one, which issue different
collectives and would hang. Which forward steps use the executor depends
only on the batch, which every TP rank receives identically, so that needs no
collective.

Four features use FlashInfer internals and run on flashinfer-python 0.7.2 or
newer (`MIN_FLASHINFER_VERSION`):

- the publishing `wo_b` GEMM: the CuTe DSL low-latency block-scaled GEMM
  kernel, and `mm_mxfp8`'s private runner and tuning config
  (`flashinfer.gemm.gemm_base`) to look up its tuned tactic;
- the fused kernel's MXFP8 layer input: the CuTe DSL MXFP8 quantize helpers
  (`flashinfer.quantization.quantization_cute_dsl_utils`,
  `flashinfer.cute_dsl.fp4_common`);
- the pinned `cute-dsl` MXFP8 GEMM tactics above 8 rows: `mm_mxfp8`'s private
  `cute-dsl` runner and workspace (`flashinfer.gemm.gemm_base`), called with
  the pinned tactic;
- the low-latency MXFP8 GEMM above 8 rows: the kernel class
  (`flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm`) built
  with a 16-, 32- or 48-row activation tile.

On an older version each logs a warning once and the executor goes without
it. `test_numerics.py` compares the first two with FlashInfer's own GEMM and
quantize bitwise, and `test_kernels.py` the last two with the stock layer; run
them when `requirements/cuda.txt` moves to a new FlashInfer.

Two more depend on FlashInfer's interface, not its version:

- the pinned low-latency MXFP8 GEMM tactics up to 8 rows (GB300) wrap that
  GEMM's `autotune_tactics`; without it, or when it does not list a pinned tactic,
  the GEMMs tune over every tactic;
- the sparse-MLA decode's persistent KV counters need the
  `multi_ctas_kv_counter_buffer` argument (0.7.1 added it); without it each
  decode allocates its own.

## Structure

The model state (`runtime/model_state.py`) builds the executor
(`runtime/decode.py`) once the weights are loaded, and tunes and compiles its
kernels (`runtime/warmup.py`) before memory profiling. For each forward, the
model asks the executor for a `DecodeStep`; the backbone runs its embedding and
Engram hashing as usual, then hands the decoder layers to the step. Only FULL
CUDA graph captures of pure decode steps of up to 384 rows get one; every other
forward runs the stock layers, which the executor does not modify.

| Package | Contents |
| --- | --- |
| `runtime/` | The executor and its per-step gates, the model state, warm-up, attention-metadata CUDA graphs |
| `boundary/` | The mHC between sublayers on the fused all-reduce + mHC kernel, and the publishing `wo_b` GEMM |
| `attention/` | The attention step: low-batch MXFP8 and BF16 projections, the row-sharded indexer, candidate overlap, shared sparse-index tables and the sparse-MLA decode |
| `moe/` | The MoE step with a deferred finalize, and the low-batch routed-expert GEMMs |
| `embedding/` | The embedding and first-layer hc streams from a replicated table |
| `engram/` | The Engram WKV ahead of the decoder layers |
| `logits/` | The low-batch LM head and the logits gather |

## Paths

| Path | Steps |
| --- | --- |
| Fused Lamport all-reduce + mHC at every sublayer boundary; the next mHC coefficients on the side stream | all executor steps (up to 384 rows) |
| Every MoE finalize deferred to the next boundary's fused kernel, in a residual-only variant before an Engram layer and after the last layer | all executor steps |
| The fused kernel also writes the attention input as MXFP8, which `fused_wqa_wkv` reads instead of quantizing again | all executor steps |
| Aux hidden-state means on the mHC side stream | all executor steps |
| Routed experts on CuTe DSL routing and `tcgen05` GEMM kernels (`moe/routed_expert_gemm.py`) | 1-192 rows |
| `fused_wqa_wkv`, `wq_b` and `wo_b` on FlashInfer's low-latency CuTe DSL MXFP8 GEMM with pinned tactics | 1-8 rows |
| `fused_wqa_wkv`, `wq_b` and `wo_b` on the low-latency GEMM with a 16-, 32- or 48-row activation tile where its shape has a wide tactic, else on FlashInfer's `cute-dsl` MXFP8 GEMM with pinned tactics | 9-48 rows |
| `fused_wqa_wkv`, `wq_b` and `wo_b` on FlashInfer's `cute-dsl` MXFP8 GEMM with pinned tactics instead of per-start autotuning | 49-384 rows |
| Compressor, indexer-weight and indexer `wk` projections on a low-batch BF16 GEMM; the index key store reads `wk`'s FP32 output | 1-8 rows |
| `wo_b` stores its output into the fused kernel's mailbox, which then only collects | 1-8 rows |
| The decode Q/KV norm and the fused `wo_a` trigger their PDL dependents (`wq_b`, `wo_b`) early, as those load their weights before their own wait | 1-48 rows |
| The embedding and the first layer's hc streams looked up from a full copy of the table on every rank (1.32 GB a GPU), without the all-reduce | all executor steps |
| Indexer query, compressor and index key on side streams, each joined at its first reader | all executor steps; the side projections up to `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` rows |
| Candidate consumers' candidate expansion and MQA-logits schedule on a side stream after the candidate source's indexer | all executor steps |
| Dense indexer rows sharded over the TP ranks, a contiguous even slice per rank (the last ones may be short or empty), and all-gathered with the executor's Lamport all-gather (`ops/cute_dsl/all_gather.py`), over NCCL above its 4 MiB a rank | uniform decode steps of at least 12 rows |
| Token top-k, and the candidate source's blocks, selected from the BF16 logits with `decode_topk` (`attention/decode_topk_cute.py`) | up to 64 rows per rank; the candidate source while its 16-CTA rows fit two waves (14 rows on a GB300) |
| Otherwise the candidate blocks selected with `cooperative_topk` on the upcast logits | up to 64 rows per rank |
| Compressed layers sharing an index source reuse one sparse-index table; the first of them builds the other KV-cache groups' tables on a side stream | all executor steps |
| Sparse-MLA decode on a CuTe DSL kernel, a thread block cluster per token with one CTA per 128 columns (`attention/sparse_mla_cute.py`) | the target's causal decode while every token's cluster is resident at once (at width 640, up to 8 requests with DSpark on a GB300) |
| Otherwise the TRTLLM-gen sparse-MLA decode, launched on a KV-counter buffer zeroed once | all other executor steps |
| Engram WKV chain on a side stream right after each layer's lookup, its two all-gathers on a Lamport all-gather of its own | all executor steps, with TP-only tables (GPU-resident or host-offloaded) and a CuTe DSL MXFP8 `wkv` |
| Attention metadata builds recorded at CUDA graph capture, one per batch shape the FULL decode graphs serve, and replayed | FULL-graph uniform decode |
| LM head on FlashInfer's CuTe DSL BF16 GEMM | 1-16 rows |
| Logits gathered with the executor's Lamport all-gather | up to 4 MiB a rank (at TP4, 32 FP32 or 64 BF16 rows) |

## Dispatch

The executor issues cross-rank collectives, so a rank that diverged would hang
rather than fall back. Every per-step gate reads only host state that all TP
ranks share (row and request counts, graph mode, the attention metadata's host
fields); nothing is read back from the GPU. The LM head and logits gather run
outside the CUDA graph (Model Runner V2 computes logits eagerly), gated by row
count and by the all-gather's size limit.
