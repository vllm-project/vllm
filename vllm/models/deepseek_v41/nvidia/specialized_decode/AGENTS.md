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
- Parallelism: TP=4 (one GB300 tray) or TP=8 (two trays of an NVL72 NVLink
  domain; each rank holds a 288-column slice of the routed experts'
  intermediate, padded to 384); no pipeline, data, expert or context
  parallelism, no micro-batching.
- Speculative decoding: DSpark, without adaptive verification.
- Attention: `FLASHINFER_MLA_SPARSE_DSV41`, fp8 KV cache, MXFP4 indexer with
  sparse logits.
- CUDA graphs: FULL graphs for decode batches only (`FULL_AND_PIECEWISE` or
  `FULL_DECODE_ONLY`).
- No `torch.compile` of the model, no sleep mode.
- No LoRA, KV transfer or batch invariance.
- DeepGEMM for the fused all-reduce + mHC. Its Lamport mailbox publishes
  over NVLink multicast when the symmetric-memory rendezvous maps one on every
  rank, and otherwise with unicast stores into each rank's mailbox
  (`VLLM_DSV41_MHC_UNICAST=1` forces the unicast path).

This check runs once, at startup. The ranks' results are combined with a MIN
all-reduce over the TP group: if any rank misses a requirement, every rank
raises, rather than some ranks running the specialized decode and others the
stock one, which issue different collectives and would hang. Which forward steps use the executor depends
only on the batch, which every TP rank receives identically, so that needs no
collective.

Four features run only on flashinfer-python 0.7.2 (`FLASHINFER_VERSION`, the
version `requirements/cuda.txt` pins). Three use FlashInfer internals:

- the publishing `wo_b` GEMM: the CuTe DSL low-latency block-scaled GEMM
  kernel, and `mm_mxfp8`'s private runner and tuning config
  (`flashinfer.gemm.gemm_base`) to look up its tuned tactic;
- the fused kernel's MXFP8 layer input: the CuTe DSL MXFP8 quantize helpers
  (`flashinfer.quantization.quantization_cute_dsl_utils`,
  `flashinfer.cute_dsl.fp4_common`);
- the pinned MXFP8 GEMM tactics: the low-latency GEMM's `autotune_tactics`.

The fourth, the sparse-MLA decode's persistent KV counters, passes the
`multi_ctas_kv_counter_buffer` argument that 0.7.1 added.

On any other version each logs a warning once and the executor goes without
it; the MXFP8 GEMMs then tune over every tactic.

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
| `engram/` | The Engram WKV ahead of the decoder layers |
| `logits/` | The low-batch LM head and the logits gather |

## Paths

| Path | Steps |
| --- | --- |
| Fused Lamport all-reduce + mHC at every sublayer boundary; the next mHC coefficients on the side stream up to 48 rows, inline beyond | all executor steps (up to 384 rows) |
| Every MoE finalize deferred to the next boundary's fused kernel, in a residual-only variant before an Engram layer and after the last layer | all executor steps |
| The fused kernel also writes the attention input as MXFP8, which `fused_wqa_wkv` reads instead of quantizing again | all executor steps |
| Aux hidden-state means on the mHC side stream | up to 48 rows |
| Routed experts on CuTe DSL routing and `tcgen05` GEMM kernels (`moe/routed_expert_gemm.py`) | 1-8 rows |
| `fused_wqa_wkv`, `wq_b` and `wo_b` on FlashInfer's low-latency CuTe DSL MXFP8 GEMM with pinned tactics | 1-8 rows |
| Compressor, indexer-weight and indexer `wk` projections on a low-batch BF16 GEMM; the index key store reads `wk`'s FP32 output | 1-8 rows |
| `wo_b` stores its output into the fused kernel's mailbox, which then only collects | 1-8 rows |
| Indexer query, compressor and index key on side streams, each joined at its first reader | all executor steps; the side projections up to `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` rows |
| Candidate consumers' candidate expansion and MQA-logits schedule on a side stream after the candidate source's indexer | up to 16 rows |
| Dense indexer rows sharded over the TP ranks and all-gathered | steps of at least 8 requests that TP divides |
| Candidate blocks selected with `cooperative_topk` | up to 64 rows per rank |
| Compressed layers sharing an index source reuse one sparse-index table; the first of them builds the other KV-cache groups' tables on a side stream | all executor steps |
| Sparse-MLA decode launched on a KV-counter buffer zeroed once | all executor steps |
| Engram WKV chain on a side stream right after each layer's lookup | up to 16 rows, with host-offloaded TP-only tables and a CuTe DSL MXFP8 `wkv` |
| Attention metadata builds recorded at CUDA graph capture, one per batch shape the FULL decode graphs serve, and replayed | FULL-graph uniform decode |
| LM head on FlashInfer's CuTe DSL BF16 GEMM | 1-16 rows |
| Logits gathered with the custom all-gather | whenever it accepts the input |

## Dispatch

The executor issues cross-rank collectives, so a rank that diverged would hang
rather than fall back. Every per-step gate reads only host state that all TP
ranks share (row and request counts, graph mode, the attention metadata's host
fields); nothing is read back from the GPU. The LM head and logits gather run
outside the CUDA graph (Model Runner V2 computes logits eagerly), gated by row
count and by the custom all-gather.
