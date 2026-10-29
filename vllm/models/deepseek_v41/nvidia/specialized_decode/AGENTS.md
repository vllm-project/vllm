# DeepSeek-V4.1 specialized decode (NVIDIA)

A decode executor for small decode batches of DeepSeek-V4.1 on GB300, and on
Hopper (`hopper/`), enabled with
`--additional-config '{"decode_backend": "latency_specialized"}'` (the
default, `auto`, keeps the stock decode). With it off, the model runs exactly as
without this package. One executor runs the target model and another the
DSpark draft model, on the target's kernels and streams.

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
- No `torch.compile` of the model, no sleep mode.
- No LoRA, KV transfer or batch invariance.
- DeepGEMM and NVLink multicast for the fused all-reduce + mHC.

On H100/H200 (SM90) the GPU and attention requirements are instead the
FlashMLA sparse attention (the default backend) with its `fp8_ds_mla` KV cache;
the rest is the same.

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
CUDA graph captures of pure decode steps of up to 384 rows (192 on Hopper) get
one; every other forward runs the stock layers, which the executor does not
modify.

The DSpark draft model registers when it is built, and the model state builds
its executor (`runtime/draft.py`) right after the target's, on the same fused
kernel, mailbox and mHC side stream; the draft model's forward asks it for a
step the same way. The model state also offers Model Runner V2 the DSpark
post-forward CUDA graphs (`vllm/v1/worker/gpu/post_forward_cudagraph.py`).
No capture size is required: a draft step without a FULL graph of at most 8
rows takes no low-batch GEMM, and the model state logs once when the capture
sizes leave no one-request draft graph.

| Package | Contents |
| --- | --- |
| `runtime/` | The target's and the draft's executors and their per-step gates, the model state, warm-up, attention-metadata CUDA graphs |
| `boundary/` | The mHC between sublayers on the fused all-reduce + mHC kernel, and the publishing `wo_b` GEMM |
| `attention/` | The attention step: low-batch MXFP8 and BF16 projections, the row-sharded indexer, candidate overlap, shared sparse-index tables and the sparse-MLA decode |
| `moe/` | The MoE step with a deferred or in-layer finalize, and the low-batch routed-expert GEMMs |
| `engram/` | The Engram WKV ahead of the decoder layers |
| `logits/` | The low-batch LM head and the logits gather |
| `hopper/` | The Hopper executor: the same runtime and steps over the FlashMLA attention, Humming GEMMs and MoE, with one function per kernel choice |

## Paths

| Path | Steps |
| --- | --- |
| Fused Lamport all-reduce + mHC at every sublayer boundary; the next mHC coefficients on the side stream up to 48 rows, inline beyond | all executor steps (up to 384 rows) |
| Every MoE finalize deferred to the next boundary's fused kernel, in a residual-only variant before an Engram layer and after the last layer | all executor steps |
| The fused kernel also writes the attention input as MXFP8, which `fused_wqa_wkv` reads instead of quantizing again | all executor steps |
| DSpark draft layers through the same steps, finalizing their MoE in the layer; the last layer's output in the fused kernel's residual-only variant | all draft executor steps |
| DSpark Markov sampling in one kernel per draft position after the bias GEMM (`vllm/v1/worker/gpu/spec_decode/dspark/fused_sample.py`) | probabilistic drafting without draft top-k, watermarking or adaptive verification |
| The whole Markov chain in one cooperative kernel per rank, the Markov head sharded by vocabulary over the TP ranks (`sharded_markov.py` beside it) | batches of up to 8 request rows |
| Sampling, the request-state update and the draft input preparation in a post-forward CUDA graph, the draft context projection beside them on a side stream | uniform decode steps without logprobs, grammar or logits processing |
| Aux hidden-state means on the mHC side stream | up to 48 rows |
| Routed experts on CuTe DSL routing, `tcgen05` GEMM and finalize kernels (`moe/routed_expert_gemm.py`) | 1-8 rows |
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

## Hopper

`HopperSpecializedDecode` runs FULL-graph pure-decode steps of up to 192 rows
(32 DSpark requests). Its boundaries run the fused all-reduce + mHC kernel up to
the stock decode's cap (`mhc_overlap_max_tokens()`, 144 rows; the stock path
enables that boundary on SM90 itself, and the executor requires it), where the
one-shot kernel still beats the stock two-shot all-reduce; above it they take
the stock all-reduce, mHC post and pre. It reuses the runtime, boundaries and
attention schedule above with these differences:

- Each kernel choice is one function in `hopper/kernels.py` (dense linears,
  `wo_a`/`wo_b`, MoE) or `hopper/sparse_mla.py` (sparse MLA). A Hopper kernel
  serves the rows it covers and reads the stock weights and caches in place,
  so larger steps and prefill keep the stock layers.
- The dense FP8 linears (`fused_wqa_wkv`, `wq_b`, the indexer's `wq_b`, `wo_a`,
  `wo_b`, the shared expert's, the Engram `wkv`) run on the CuTe DSL kernel
  (`hopper/dense_gemm_cute.py`, up to 48 rows) or tuned Humming (up to 192),
  as `hopper/dense.py`'s table picks, all with PDL; `wo_b`'s CuTe kernel
  publishes into the fused all-reduce's mailbox.
- The MoE runs one persistent CuTe DSL launch (`hopper/moe_cute.py`, host in
  `hopper/moe.py`) on the stock Humming W4A8 experts: routing from the stock
  router's logits, both expert GEMMs and the shared expert, up to 144 rows.
  It leaves the top-k reduction and shared add to the next boundary's
  `AllReduceMHC.finalize`; every layer shares one workspace (about 230 MB).
- The executor compiles every kernel, and JIT-compiles Humming's, before CUDA
  graph capture.
- Memory: the stock SM90 `wo_a` keeps only the BF16 weight it dequantizes at
  load, so the executor builds an exact FP8 copy in the Humming layout at
  construction: about 8.7 MB a layer, 350 MB a GPU at TP4. Larger steps keep
  the stock BF16 bmm.
- The residual-only boundaries also take a finalized sublayer output (from a
  MoE layer the low-batch MoE does not take).
- The sparse MLA shares one FlashMLA plan per layer type across the SWA
  KV-cache groups and one global top-k table per index source. Uniform steps
  of up to 12 rows (one or two DSpark verify requests) run the small-batch
  kernels instead (`hopper/mla_small.py`, `hopper/mla_cute.py`): SWA unions per
  KV-cache group on a side stream, one compressed prep per index source from
  the indexer's local top-k, and a decode plus combine per layer, within one
  BF16 ulp of FlashMLA at the output's scale.
- The MoE kernel takes the layer's expert count and top-k (up to 384 and 6).
- Up to 8 rows on the small-batch MLA, the attention chain's small kernels
  are early-triggering CuTe DSL ones (`hopper/attn_chain.py`): q/kv RMSNorm
  with the KV insert, the query RoPE, and the MLA combine with the inverse
  RoPE writing the `wo_a` input.
- Index sources up to 48 rows run the fused FP8 indexer
  (`hopper/fused_indexer.py`): logits with all verify rows per index-K tile and
  the candidate bitmap, then the stock `cooperative_topk`.
- `wq_b`, `wo_b` and `wo_a` multicast X within clusters at 6-16 rows (`cn`).
- The CuTe DSL kernels (`hopper/*_cute.py`, PTX helpers in `*_ptx.py`) began
  as out-of-tree generated prototypes and are kept specialized to the
  configurations their host modules compile; edit them in place and keep each
  host's `_compiled` arguments the only switches.
- The DSpark draft runs `HopperDraftDecode` (`hopper/draft.py`): the same
  attention and MoE choices on the draft's own FlashMLA tables, its MoE (128
  experts, top-3) finalized in the layer by the routed-expert finalize kernel,
  and `main_proj` on tuned Humming up to 48 rows. The vocabulary-sharded
  Markov sampler needs SM100; the fused one runs.
- No low-batch GB300 kernels, MXFP8 layer input, row-sharded indexer or
  candidate overlap (the FP8 indexer masks dense logits).
- The Engram WKV runs ahead on its side stream.
- Metadata builds are recorded for the FlashMLA builders, up to the cap.

## Dispatch

The executor issues cross-rank collectives, so a rank that diverged would hang
rather than fall back. Every per-step gate reads only host state that all TP
ranks share (row and request counts, graph mode, the attention metadata's host
fields); nothing is read back from the GPU. The LM head and logits gather run
outside the target's CUDA graph (Model Runner V2 computes logits eagerly),
gated by row count and by the custom all-gather. On uniform decode steps with
DSpark, they, the sampling and the draft input preparation are captured in a
post-forward CUDA graph of their own.
