# Qwen4Exp specialized decode (NVIDIA)

A decode executor for small decode batches of Qwen4Exp (Qwen3.8-Flash-Next) on
GB300, enabled with `--additional-config '{"decode_backend": "latency_specialized"}'`
(the default, `auto`, keeps the stock decode). With it off, the model runs
exactly as without this package. The executor runs the target model; in the
MTP draft model (`../mtp.py`) only the passes that take the fused front (its
two input norms and two projections, `runtime/draft_front.py`, which then runs
the draft layer and the final mixer as the executor runs a target layer) and,
with `use_local_argmax_reduction`, its token choice (`logits/choice.py`)
differ from stock. The target's logits processor is `logits/processor.py`'s, which
gathers its logits at TP4 with `logits/gather.py` whenever the setting is on;
the draft model's is too when `use_local_argmax_reduction` is set. At TP4 the
executor's steps gather the PLE layer's lookup with `ple/gather.py`.

## Scope

`specialized_decode_config` (`__init__.py`) reads the setting once, when the
target model is built. Any deployment other than the one below raises at
startup on every TP rank with the missing requirements:

- GPU: GB300 (SM103).
- Model: hyper-connections of 4 streams of 2560 values with low rank 320
  (Qwen3.8-Flash-Next), the shape the read kernels are built for.
- Parallelism: TP=1 or TP=4, with or without expert parallelism; no pipeline,
  data or context parallelism, no micro-batching, no hyper-connection or MoE
  sequence parallelism.
- Speculative decoding: MTP.
- CUDA graphs: FULL graphs for decode batches.
- No `torch.compile` of the model, no sleep mode, no runtime weight updates, no DeepGEMM MegaMoE experts.
- No LoRA or batch invariance; no KV transfer but vLLM's in-tree `OffloadingConnector` with an in-tree CPU spec (what `--kv-offloading-size` sets with the native backend).

One more setting is not required but serves the draft passes: with
`"use_local_argmax_reduction": true` in `--speculative-config`, Model Runner V2
asks the draft model for its tokens, which `logits/choice.py` chooses on the
vocabulary shard. Without it the token choice is neither built nor warmed up,
the draft passes keep the stock choice (a logits all-gather and
`greedy_argmax`, 15 to 65 us a pass at TP4), and the check logs that once at
startup. The target's logits all-gather at TP4 (`logits/gather.py`) does not
depend on it: it is on whenever the setting is.

`"draft_head_fp8_prefilter": true` in `--additional-config` (default false)
builds the exact FP8-prefiltered draft head (`logits/draft_head.py`) at TP1:
draft passes of up to 8 rows find their greedy tokens from an FP8 copy of the
LM head and rescore only the entries that could win with the BF16 weight in
cuBLAS's order: the stock head's tokens, about 64 us sooner a pass at one
row and 51 us at eight (GB300, 183 us stock). The
copy and its statistics take 0.65 GB, which memory profiling counts, so the KV
cache shrinks by as much (GB300, TP1, `--max-num-seqs 64`: 4,006 instead of
4,022 blocks of 1,600 tokens) while only decode steps of at most 8 requests
use it; turn it on for low-concurrency TP1 deployments. It needs TP=1,
`use_local_argmax_reduction` and a drafter that shares the target's BF16 head
(no `draft_lm_head_quantization`), decided from the config alone, and raises
at startup otherwise; the model state also raises when the head's stock GEMM
is not `F.linear` on GB300 (a quantized or skinny-GEMM head).

The ranks' results are combined with a MIN all-reduce over the TP group's CPU
group: if any rank misses a requirement, every rank raises, rather than some
ranks running the executor and others the stock decode, which issue different
collectives once the executor has its own.

The executor takes decode steps of up to `max_decode_rows` rows: 64 requests at
TP1 and TP4 (`MAX_DECODE_REQUESTS`), of `num_speculative_tokens + 1` rows
each; 256 rows with three draft tokens.

## Structure

The model state (`runtime/model_state.py`) builds the executor
(`runtime/decode.py`) once the weights are loaded. For each forward, the model
(`Qwen4ExpForCausalLM.plan_decode`, also called by the multimodal wrapper) asks
the executor for a `DecodeStep`; the backbone embeds the tokens as usual, then
hands the decoder layers and the final mixer to the step. The executor's loop
(`SpecializedDecode.run`) is the stock one (the layer loop and final mixer of
`Qwen4ExpModel.forward`, the non-sequence-parallel
`Qwen4ExpDecoderLayer.forward`) with the same submodule calls; only the
hyper-connection reads, the sublayer boundaries, the GDN layers' core, the QSA
layers' sparse attention (without the zero fill of the step's rows, and the
row attention in steps of 128 rows and more at TP1), at TP4 the PLE layer's
lookup (`runtime/ple.py`) and, in steps of up to 32 rows, the QSA indexers' select differ.
A draft pass that takes the MTP draft front runs the draft layer and the final
mixer the same way (`DraftFront.run_pass`): the reads, the GEMM plans, the
QSA attention and the sublayer boundaries, with the draft layer's own
boundary kernel instance and mailbox.

| Package | Contents |
| --- | --- |
| `runtime/` | The executor and its per-step gate, the model state, the executor's GDN layers (`gdn.py`) and their record/commit state protocol (`gdn_records.py`), the sublayer boundaries (`boundary.py`), the MTP draft front (`draft_front.py`), the decode GEMM plans (`gemm.py`), the QSA layers' attention without the output fill of the step's rows, and with the row attention in large steps, and their indexers' CuTe select in small steps (`qsa.py`), the PLE layer's gathered lookup at TP > 1 (`ple.py`), the row split of large TP4 steps (`split.py`), the TP1 routed MoE path (`moe.py`), the replicated token embedding at TP > 1 (`embedding.py`), the replayed attention metadata and n-gram inputs of uniform decode steps (`decode_metadata.py`) |
| `hc/` | The hyper-connection read kernels (CuTe DSL) and the executor's read |
| `boundary/` | The sublayer boundary kernel (CuTe DSL) |
| `gdn/` | The fused GDN decode core, the staged-weight GDN projections and the state commit of the core's record/commit protocol (CuTe DSL) |
| `mtp/` | The MTP draft passes' fused input norms (CuTe DSL) |
| `qsa/` | The QSA sparse attention with one CTA per row and KV head, and the QSA indexer's decode top-512 block select with the index expansion, one cluster a row (CuTe DSL) |
| `logits/` | The MTP draft passes' token choice and FP8-prefiltered draft head, the target's logits all-gather and its rank-order variant for the row split's streams (CuTe DSL), and the logits processor that binds them |
| `ple/` | The PLE layer's row gather and dequantization at TP > 1 (CuTe DSL) |
| `moe/` | The routed MoE front of TP1 steps of up to 32 rows: top-10 routing, routing metadata and NVFP4 quantization in one kernel (CuTe DSL) |

## Paths

| Path | Steps |
| --- | --- |
| The stock layer loop (PLE prefetch one layer ahead, QSA attention, MoE, PLE combines, deepstack) and the final mixer's MTP hidden-state copy | all executor steps |
| Hyper-connection reads: stock norm or combine-norm (dependents launched right after its wait), `hc_down`, `hc_up_mix` | up to 64 rows (with the tensor-core reads below `TC_MIN_ROWS`), in executor steps and in the draft passes that take the fused front (whose first read combines the projected embedding without injection logits) |
| Tensor-core hyper-connection reads (`hc/read.py`): the same norms, then `hc_down_tc` (tcgen05 UMMAs of 128 token rows into tensor memory; 16-CTA clusters split K and reduce-scatter the partials through distributed shared memory; the `hc_down_silu` epilogue) and the fused up projection and gate mix of `../ops/hc.py` (`hc_up_gate_mix`, CUDA, FP32 gate). One TP1 site at 128 rows: 20.8 instead of 26.7 us (stock read; GB300 microbenchmark) | TP1 executor steps of `TC_MIN_ROWS` (52) to 128 rows; a device or build without tcgen05, the fused kernel, the CTAs' shared memory or 16-CTA clusters keeps the bounds of the rows around this one (`use_tensor_core_reads`, logged once); draft passes keep 64 rows |
| Hyper-connection reads: stock `GatedResidual.mix` / `combine_and_mix` | TP1 executor steps above 128 rows (above 64 without the tensor-core reads), the reads of TP4 executor steps above 64 rows that do not split (before the split's first read, or a step the split does not take), draft passes above 64 rows, and every draft pass outside the fused front |
| Sublayer boundaries: the block's projection or MoE runner without its all-reduce, then `SublayerBoundary` (one-shot Lamport all-reduce over its own symmetric-memory mailbox, the MoE shared-expert gate and routed add, the combine and the next read's norm), then the read's projections (the kernel ones up to the read kernels' bound, the stock ones above) | TP4: attention, GDN and MoE blocks; TP1: MoE blocks; except a MoE block before a PLE layer and every MoE block of a step with deepstack embeddings. The same for the draft layer's attention and MoE blocks in the draft passes that take the fused front, with the draft layer's own kernel instance (built at the end of the draft's `load_weights`; its MoE block is checked again at each capture, after the quantization methods processed the weights) |
| Folded routed finalize: FlashInfer's TRT-LLM NVFP4 experts stop after GEMM2 (`post_gemm2_hook`, with an unwritten `output` of the finalized shape so that FlashInfer's autotuner keeps the tactics tuned for the finalized call), and `SublayerBoundary.moe_folded` runs their top-10 reduction in the finalize kernel's order, bitwise equal (`runtime/boundary.py` states the ordering invariants; graph captures check that the boundary node follows GEMM2 and that no finalize ran) | the MoE boundaries above whose experts are FlashInfer's TRT-LLM NVFP4 monolithic experts with top-10 routing, in steps of 32 rows or more (`FOLD_MIN_ROWS`, static per captured shape); smaller steps keep FlashInfer's finalize kernel and the unfolded MoE boundary, since there the shared expert, not the routed chain, ends last |
| Routed MoE front (`runtime/moe.py` states the ordering invariants): one CuTe DSL kernel (`moe/front.py`) replaces vLLM's NVFP4 quantization of the layer input and FlashInfer's routing kernels (the top-10 of the BF16 logits and the softmax over the ten, in FlashInfer's arithmetic; the routing metadata the expert GEMMs read), then FlashInfer's expert GEMMs run on that record through its prepared-routing body with the tactic the autotuner holds for the stock call (its fallback included), finalized or folded as above; bitwise the stock routed output | TP1 MoE boundaries above in steps of up to 32 rows whose stock GEMMs use the 8-row token tile, for FlashInfer's monolithic TRT-LLM NVFP4 experts with the model's routing (`unsupported`, logged once); the draft layer keeps the stock MoE |
| Sublayer boundaries: the layer's own all-reduce, the stock shared-expert gate, then the next read | the other blocks |
| Row-split reads (`runtime/split.py` states the invariants): from the MLP read of the last PLE layer (layer 0's without one) through the final mixer every rank keeps its quarter of the rows of the streams and injection logits; `SublayerBoundary`'s split call sends each row of the partial block output to the rank that owns it, which sums the four in rank order (the one-shot sum's bits) and combines and normalizes its rows; `hc_down` of the owned rows; `HcUpMixGather` (their up/mix, then a Lamport all-gather of the block input in the same kernel); after the final mixer `RowsAllGather` gathers the owned streams into the MTP hidden buffer. Bitwise the replicated kernel reads, with negative zeros as positive zeros in the gathered tensors. The boundary-and-read sites take 14.3 instead of 19.8 us at 64 rows and 13.2 instead of 13.9 us at 32 (TP4, GB300); above 64 rows a rank still reads at most 64 (`OWNED_MAX_ROWS`), where the replicated reads would be the stock ones | executor steps at TP4 of 32 to 256 rows, multiples of 4, without deepstack embeddings, when the boundary kernel serves the attention blocks and every MoE block from the split on (`split_unsupported`, logged once); smaller steps, TP1 and the draft passes read replicated |
| GDN layers, staged: `GdnProjection` for `in_proj_qkvz` and `in_proj_ba` in one launch (each CTA copies its weight share into shared memory before its PDL wait; no side stream), `gdn_decode_core`, then `GdnProjection` for `out_proj`, which writes the rank's BF16 partial output for the next read's boundary kernel to reduce; within BF16 rounding of an FP32 reference, not bitwise (the projections' k order); the core runs the record/commit state protocol as in the other fused steps | the fused core's steps at TP > 1 of up to `STAGED_MAX_ROWS` (16) rows, when every GDN layer's three projections are unquantized BF16 without bias and their weights fit one CTA per SM (`staged_projection_unsupported`, logged once) |
| GDN layers, other fused steps: the layer's `in_proj_qkvz`, its `in_proj_ba` on a side stream, `gdn_decode_core` (convolution, gated delta rule and gated RMSNorm in one kernel; states bitwise equal to the stock core's, outputs within its last bit), the layer's `out_proj`; the core runs the record/commit state protocol (below) unless mixed batches take FULL CUDA graphs (`cudagraph_mode` FULL), where it keeps vLLM's | speculative decode steps of `num_speculative_tokens + 1` rows per request, at most `MAX_CORE_TOKENS` (8; the stock fused MTP path takes up to `MAX_FUSED_GDN_MTP_TOKENS`), FP32 recurrent state; padding rows of the core output hold zeros or earlier finite outputs (an executor buffer zeroed at build) |
| GDN layers: the stock layer | other steps; every step with a BF16 recurrent state, more than `MAX_CORE_TOKENS` rows per request or a layer the fused core does not cover (logged once at startup) |
| QSA sparse attention: `qsa_row_attention` (one 128-thread CTA per row and KV head streams the row's selected K/V rows through three `cp.async` stages, `mma.sync` scores and P V with a per-warp online softmax, the warps merged in shared memory, the gate applied at stock's rounding point; no FP32 partials or merge kernel) without stock's zero fill of the output (the kernel writes every value of the step's rows, padding rows as stock's zeros; only rows past the step's are zero-filled), `runtime/qsa.py` swapping the layers' `impl` while the step is captured; within stock's error against an FP32 reference, not bitwise | launches of `MIN_CTAS` (256) CTAs and more (rows times the rank's KV heads: 128 rows at TP1, 256 at TP4), static per captured shape: the executor's (TP1 at 32 requests and more, TP4 at 64) and the draft layer's in the draft passes that take the fused front (the first pass of such a step), for BF16 main K/V caches the kernel takes (`row_attention_unsupported`, logged once) |
| QSA sparse attention: the stock Triton split-K kernel and merge without stock's zero fill of the step's rows (the split-K kernel at one split, the merge at more, writes every value of the step's rows; only rows past the step's are zero-filled), `runtime/qsa.py` swapping the layers' `impl` while the step is captured; bitwise stock's | smaller launches of the executor and of the draft passes that take the fused front, and launches the row attention does not take, for BF16 main K/V caches |
| QSA indexer, block select: the stock fused prepare and score kernel, then `qsa_select_expand` (one cluster of 4 to 16 CTAs a row: a radix select of the 512th score over the cluster's registers, distributed-shared-memory histograms, each selected block written as one 16-byte store of its 4 tokens, the open block's causal tail and the count) in place of the stock top-k and index expansion, `runtime/qsa.py` swapping the indexers' `forward` while the step is captured; the stock set of blocks (ties at the 512th score: lowest block index first, or in CTA, thread and key order when more than 4,096 scores share its quarter octave) and the stock packed row bitwise for rows of at most 512 visible blocks, the blocks in its own order otherwise | pure decode steps of up to `SELECT_ROWS` (32) rows of the executor and of the draft passes that take the fused front, static per captured shape, for indexers with the fused prepare and 512 blocks of 4 tokens and a model length of at most 262,144 tokens; the MTP steps that reuse step 0's selection keep stock |
| QSA indexer, block select: the stock score kernel, top-k and index expansion | larger steps, layers with FP8 main K/V caches, and every forward outside the executor and the fused front |
| QSA sparse attention: the stock forward (zero fill, split-K kernel and merge) | FP8 main K/V caches, and every forward outside the executor and the fused front |
| MTP draft front: `front_norm` (both input norms, `1 + w` in FP32), then `fc_embedding` and `fc_hidden` on full weight copies (no all-gathers; skinny CuTe GEMMs at 1 and 4 rows on GB300, `F.linear` otherwise) | FULL-graph draft passes of pure decode steps of up to `max_decode_rows` rows; at TP > 1 the full copies (26 MB per rank) are filled from the checkpoint tensors by the projections' wrapped weight loaders |
| The layers' BF16 projections (GDN outside the staged steps, QSA, MoE router, shared expert): the skinny CuTe GEMM plans of `QWEN4_EXP_DECODE_GEMM_PLANS` (`runtime/gemm.py`) at the row counts they list, the layer's own GEMM (the stock table's plan or `F.linear`) otherwise | all executor steps, and the draft layer in the draft passes that take the fused front |
| MTP draft front: stock (`GemmaRMSNorm` pair, column-parallel projections with all-gathers) | every other draft forward; every forward when the front refuses the deployment (quantized projections, a replica not loaded from the checkpoint, or weights never bound because the loader skipped the draft's `load_weights`: dummy weights, the weight cache, sharded state; logged once) |
| MTP draft token choice: `TokenChoice` on the rank's vocabulary shard of the draft head's BF16 logits (one kernel; at TP4 the ranks exchange one tagged 8-byte record per row over symmetric memory), the token `greedy_argmax` returns on the gathered logits | draft passes of 1 to `min(max_num_seqs, MAX_DECODE_REQUESTS)` rows (64 at TP1 and TP4; fewer than the SMs, which its liveness needs) when the speculative config sets `use_local_argmax_reduction` (else it is not built and the passes run the stock `compute_logits` + `greedy_argmax`); the stock local argmax for larger passes, a scaled or soft-capped head, `head_dtype` other than BF16, added vocabulary, or non-BF16 logits (logged once) |
| MTP draft head: `DraftHead` (quantize the rows to two E4M3 code rows, cuBLASLt FP8 GEMM against the head's E4M3 copy, a rigorous bound per entry, then the candidates whose bound reaches the best lower bound rescored with the BF16 weight by `mma.sync` chains that reproduce cuBLAS's sums bit for bit); the first index of the largest BF16 logit, as the stock head and `TokenChoice` | TP1 draft passes of 1 to 8 rows with `draft_head_fp8_prefilter` (built by the model state; the drafter shares the target's head); the token choice or the stock path for larger passes and every other head |
| PLE lookup at TP > 1: the stock side-stream lookup one layer ahead, then `PleRowGather` (`runtime/ple.py` swapping the table's `finalize_prefetch` and the layer's dequant while the step is captured): one kernel of at most the measured resident CTAs joins nothing but the side stream's lookup; each row's owner (the rank whose vocabulary range holds its id) dequantizes it and stores it into its output and into the row's slots of every peer's two-generation symmetric-memory mailbox, the other ranks poll their slots; the stock rows (`rows.to(bf16) * scale.to(bf16)`) bit for bit with negative zeros as positive zeros, the same on every rank; its mailbox is two generations of `max_decode_rows` tokens times the PLE width in BF16 a rank (256 tokens at TP4) | executor steps at TP4 of 1 to `max_decode_rows` tokens, for pinned-host FP8 PLE tables split over the TP group (`gather_unsupported`, logged once) |
| PLE projection after the gather: `kv_proj` (`PaddedProjection`, `runtime/ple.py`) multiplies the first 16 rows of the gather's buffer and keeps the step's rows (the other rows hold zeros or earlier steps' rows; an output row depends on its own input row only), so cuBLAS takes one kernel instead of split-K and a reduction (11.6 against 13.4 us); within BF16 rounding of the stock projection, not bitwise (cuBLAS sums K in another order) | the gathered steps above of fewer than 16 tokens (TP4 at 1 to 3 requests); the projection's own GEMM for larger steps |
| PLE lookup: the stock finalize (int8 all-reduce over the ETP group, copy) and dequant | TP1, other tables, and every forward outside the executor |
| Token embedding at TP > 1: `ReplicatedEmbedding` (`runtime/embedding.py`): the shards all-gathered into one BF16 table on every rank when the executor is built (the module's shard becomes a view of it: 0.95 GB more a rank for Qwen3.8-Flash-Next, which memory profiling counts), and the module's forward routed to the stock fused lookup over the full vocabulary, without the all-reduce; the stock rows bit for bit except that negative zeros stay negative | every lookup captured into a CUDA graph at TP4 with the setting on: the target's and, through the shared module, the MTP drafter's; unquantized BF16 embeddings without added vocabulary (`replication_unsupported`, logged once) |
| Token embedding: the stock vocab-parallel lookup and all-reduce | eager forwards (prefill, mixed batches, profiling), and TP1 |
| Target logits all-gather: `LogitsAllGather` (one kernel of at most the measured resident CTAs: occupancy per SM times the SM count, one per SM if the cubin is missing; Lamport exchange over a two-generation symmetric-memory mailbox, written in vocabulary order; about 182 MiB a rank at TP4 for 256 rows) | TP4 calls of 1 to `max_decode_rows` BF16 rows whenever the setting is on, with or without `use_local_argmax_reduction`; the stock all-gather otherwise and at TP1 |

The model state runs every variant of the read kernels, the boundary kernel,
every schedule of the fused GDN core, the staged-weight GDN projections, the QSA row attention, every decode GEMM plan
and the PLE row gather once (`SpecializedDecode.warm_up`), then the GDN state commit of every step size
(`GdnRecords.warm_up`), right after building the executor, before memory profiling: its CUDA graph captures of the largest
decode steps already take the executor, and no capture may compile a kernel.
The draft front builds the draft layer's boundaries and warms up its kernels and the draft layer's GEMM plans, row attention, reads and boundary kernel at the
end of the draft model's `load_weights` instead: Model Runner V2 builds the model state from the target
model alone, after loading both models, so that is the draft's last hook
before the model state, memory profiling and the speculator's captures. The
token choice is built and warmed up there too, after the draft front (only
with `use_local_argmax_reduction`); the logits all-gather with the executor.
With `draft_head_fp8_prefilter`, the model state builds and warms up the
FP8-prefiltered draft head right after the executor (`bind_draft_head`), from
the target's LM head, which the drafter shares by then.

## Host metadata

Before every FULL graph replay the runner rebuilds the attention metadata on
the host, which rewrites the persistent buffers the graph reads. For uniform
decode steps (every request schedules its `num_speculative_tokens` drafts plus
one, no prefills, no adaptive verification) the model state instead replays a
CUDA graph of the stock build (`MambaHybridModelState.prepare_attn`), recorded
per `(requests, padded requests, tokens, padded tokens)` key with the inner
CUDA graphs, and returns the metadata objects recorded with it after setting
`max_seq_len`; the PLE n-gram inputs of such a step take one Triton kernel.
Every other step, and `preprocess_state`, run the stock code.
`runtime/decode_metadata.py` documents the invariants: why the host values are
constant per key, what a replay rewrites and which persistent buffers it
relies on. `additional_config={"decode_metadata_check_steps": N}` checks the
first N replays of every key against the stock build run from the same buffer
state and raises on any difference;
`tests/models/qwen4_exp/specialized_decode/test_metadata.py` checks every
captured size.

The replay overlaps the forward graph (`additional_config` setting
`decode_metadata_overlap`, default true). A key replays on a side stream of
`DecodeMetadataGraphs` when the FULL forward graph of its padded token count
was captured with the executor's waits (`CaptureWaits`: an external event wait
for `gdn_ready` before the first GDN layer, for `ready` before the first PLE or
QSA layer and at the end at the latest) and no FULL capture of that count ran
the stock layers. Each recording records `gdn_ready` right after the last GDN
group's build; `ready` is recorded after the replay. These invariants keep the
stock order:

- I1 (fork): the side stream waits for the runner's stream after the step's
  prep and `preprocess_state` are enqueued, so the replay sees them and runs
  after every earlier reader of what it writes (the previous step's forward
  graph and drafts).
- I2 (join): the forward graph waits for `gdn_ready` before its first GDN
  layer and for `ready` before its first PLE or QSA layer and its end; the GDN
  metadata are final at `gdn_ready` (recording checks it); the forward graph's
  work before the waits touches nothing the replay touches.
- I3 (completion): everything after the forward graph runs after `ready`; a
  stock build or a replay on the runner's stream first waits for `ready`
  (`settle`) should no forward graph have followed an overlapped replay.
- I4 (lifetime): the step's request indices are `record_stream`ed onto the
  side stream; recordings are dropped only after it is synchronized.
- I5 (memory): recordings capture on torch's capture stream, the forward
  graphs on the runner's, so the shared graph pool never gives one's memory
  to the other's temporaries.

A graph's event waits bind, at its launch, to the events' last records before
it: CUDA's `cudaEventRecord` rule (a wait uses the most recently recorded state
at the time of the call that enqueues it, unaffected by later records) applied
to the launch that enqueues the wait nodes. So step k's forward graph sees step
k's replay even when the host has launched steps k + 1 and k + 2 (whose side
replays follow step k's forward graph: a wait bound when it runs would
deadlock). `test_metadata.py`'s
`test_a_graph_wait_binds_to_the_records_before_its_launch` checks this on the
GPU with the host far ahead; `test_metadata_protocol.py` checks I1-I4 over
every interleaving of the two streams with three steps enqueued, on what the
module enqueues, and catches each invariant's defect.

## GDN state protocol

In vLLM's speculative-decode protocol each GDN layer's verify writes the FP32
recurrent state after every verified row to that row's slot (3 MiB a request
and layer at TP1, 768 KiB at TP4), although only the last accepted row's slot,
and on a prefix-caching block boundary one more, is read again. In the
executor's FULL graphs the fused core instead writes small per-row records
(keys, decays and delta-rule corrections: 129 KiB a request and layer at TP1,
32 KiB at TP4) and no state; the model state marks a step that replays such a
graph in `prepare_attn` and, in `postprocess_state` before the stock
postprocess, launches the commit (`gdn_state_commit`, one eager launch for
all layers), which folds the accepted rows into those slots bit for bit as
the stock protocol writes them. The records take no memory of their own: the
verify writes two copies into two of the request's other `q - 1` speculative
slots (which hold nothing anyone reads until the next verify), at the start
of each key head's state rows, and the commit reads one it does not write.
The commit writes slot `a - 1` and, in prefix-caching `align` mode, the
boundary row's slot; the verify cannot know `a`, but it knows the boundary
row's slot from the runner's `seq_lens` and the mamba block size, so it
avoids that slot, and `a - 1` is at most one of the two. Only a header per
request and layer (its rows, slots and copy slots) has its own buffer (63
KiB at TP1 with 64 requests). Two copy slots exist when `q - 1 >= 2` (3 in
`align` mode, with mamba blocks of at least `q - 1` tokens); otherwise the
GDN layers keep vLLM's protocol. `runtime/gdn_records.py` states the
invariants; captures raise if a FULL graph size holds the records core in only
some GDN layers or in only some of its graphs, binding the caches raises if a
state slot cannot hold the records, `prepare_attn` raises if the runner's
`seq_lens` buffer moves after the graphs captured it, an uncommitted step
raises at the next `prepare_attn`, and the commit traps on a request without
a copy it does not write.
`test_gdn_records_protocol.py` checks the protocol on a model of the runner's
steps, with injected defects; `test_gdn_records.py` the kernels and real steps
of the GDN layer on the GPU.

## Dispatch

Only FULL CUDA graph captures of pure decode steps run the executor. `plan()`
returns a step only while a CUDA graph capture is in progress and not a
PIECEWISE one, for 1 to `max_decode_rows` rows, and when the attention metadata
of every KV-cache group of the layers reports no prefills: `num_prefills == 0`
for GDN, the PLE short convolution and the QSA indexer caches, and
`max_query_len <= num_speculative_tokens + 1` for the QSA main cache. Every
other forward (prefill, mixed, eager, PIECEWISE, the eager run before each
capture) runs the stock layers.

The executor issues cross-rank collectives (the boundary kernel's exchange,
and at TP4 the token choice's, the logits gather's, the PLE gather's and the
row split's exchanges), so a rank that diverged would stall the others rather than fall
back: their polls give up after 30 s, record the stall
(`SublayerBoundary.stall_report`, `TokenChoice.stall_report`,
`LogitsAllGather.stall_report` (also for `RowsAllGather`),
`PleRowGather.stall_report`, `HcUpMixGather.stall_report`) and trap, and the
step raises a CUDA error. Which calls the token choice and the gathers take is
decided from shapes and dtypes only, the same on every rank. Every per-step gate reads only host state that
all TP ranks share (row counts, graph mode, the attention metadata's host
fields); nothing is read back from the GPU. The ranks compare the MoE blocks the
boundary kernel serves when the executor and the draft front build their
boundaries, and all raise if they differ. `runtime/draft_front.py` ("Exchange
order") states why the draft layer's instance keeps the exchange's order.

## PLE gather protocol

[ple/AGENTS.md](ple/AGENTS.md) states the PLE row gather's invariants and tests.

## Outside this package

Everything the specialized decode needs lives here, and its tests in
`tests/models/qwen4_exp/specialized_decode/`, except these hooks and stock
changes:

- `../model.py`: `Qwen4ExpModel.forward` hands a planned step to
  `DecodeStep.run` after the embedding; `Qwen4ExpForCausalLM` reads the
  setting, picks the model state, builds the executor
  (`build_specialized_decode`, which also binds the logits all-gather) and
  plans each forward (`plan_decode`); with the setting on, its logits
  processor is `SpecializedDecodeLogitsProcessor`; the multimodal wrapper
  delegates to it.
- `../mtp.py`: the draft model builds the `DraftFront`, hands it the passes it
  takes (`DraftFront.run_pass`) and binds it at the end of `load_weights`,
  then, with `use_local_argmax_reduction`, binds the token choice
  (`bind_token_choice`) to its `SpecializedDecodeLogitsProcessor` (without
  it the draft keeps the stock `LogitsProcessor`). `Qwen4ExpMTP.get_top_tokens` is stock
  code for every decode backend: it lets the speculative config's
  `use_local_argmax_reduction` serve Qwen4Exp's MTP (before, that option
  raised at startup), through the stock vocab-parallel argmax without the
  setting.
- `../ops/hc.py`: `hc_combine_norm(launch_dependents_early=True)`, an opt-in
  parameter of the stock Triton combine-norm that the read kernels set
  (`hc/read.py`); tested in `test_hc_ops.py`.
- `../../common/qsa_indexer.py`:
  `QSAIndexerMetadataBuilder.update_draft_decode_metadata`, the QSA indexer
  caches' in-place draft metadata refresh. It is stock code for every decode
  backend: Model Runner V2 records all draft decode steps in one CUDA graph
  only when every draft attention backend refreshes its metadata in place
  (`supports_draft_decode_metadata_update`), else it rebuilds the metadata of
  each draft step on the host, so moving the refresh here would change the
  default decode. Tested in `test_qsa_draft_metadata.py`.
