# MiniMax-M3 specialized decode (NVIDIA)

Kernels for small decode batches of MiniMax-M3.1-Flash on B200 and B300,
enabled with `--additional-config '{"decode_backend": "latency_specialized"}'`
(the default, `auto`, keeps the stock decode). With it off, the model runs
exactly as without this package. A decode executor runs the decoder layers of
FULL-graph decode steps of 1-32 rows (up to four requests with the seven-token
DSpark draft) with a low-batch sparse attention core (index scores, top-k,
sparse attention and o_proj, after the stock qkv projection and KV insert) and
a low-batch FFN half of the MoE layers; the qkv projection and insert, the
all-reduce, the dense layers and the MTP draft keep their stock modules.

## Scope

`specialized_decode_config` (`__init__.py`) reads the setting once, when the
model is built. Any deployment other than the one below raises at startup on
every TP rank with the missing requirements:

- GPU: SM100 family (B200, B300), one code path.
- Runner: Model Runner V2.
- Parallelism: tensor parallel only; no pipeline, data, expert or context
  parallelism, no micro-batching.
- Layers: hidden size 6144, 128 experts, top-4, 768 intermediate columns per
  rank for the routed and the shared expert (MiniMax-M3.1-Flash at TP4), no
  MegaMoE.
- CUDA graphs: FULL graphs for decode batches only (`FULL_AND_PIECEWISE` or
  `FULL_DECODE_ONLY`).
- No `torch.compile` of the model, no sleep mode, no LoRA or batch invariance.
- The fused MoE finalize + all-reduce + norm (`ops/finalize_allreduce_norm.py`),
  which needs NVLink multicast on every rank.

Every rank gives the same answer: the other checks read only the configuration,
and `supports_fused_moe_finalize` decides the multicast probe over the TP group
(an all-ranks reduction), so either every rank raises or none does.

Two more checks follow, and they also raise on every rank, because the
configuration decides both:

- When the model is built, at least one MoE layer must leave its top-k finalize
  to the next layer's fused kernel. Only those layers get the low-batch FFN;
  the last MoE layer finalizes itself and stays stock.
- Once the weights are processed, each of those layers must be able to take
  the FFN (`moe.unsupported_reason`):
    - TRT-LLM-gen monolithic NVFP4 experts with per-token activations;
    - a FlashInfer MXFP8 shared expert;
    - the SwiGLU-OAI constants the kernels are compiled with.

The low-batch attention needs no check of its own: a sparse attention layer
without 16 query heads on one KV head and one index head of 128, NVFP4 KV and
index caches, 16 top-k blocks, unit KV global scales and a FlashInfer
`cute-dsl` MXFP8 o_proj gets no `LowBatchAttention`
(`attention.build_low_batch_attention` returns None) and keeps its stock
modules. The configuration decides this, so every rank agrees.

## Structure

| Path | Contents |
| --- | --- |
| `__init__.py` | The setting, the startup checks, the capture gate |
| `runtime/decode.py` | `SpecializedDecode` (`plan`, the step's attention metadata, the layer loop) and `DecodeStep` |
| `attention/__init__.py` | `build_low_batch_attention`, `LowBatchAttention` (`accepts`, `run`), `prepare_step`, `MAX_ROWS` |
| `attention/indexer_cute.py` | The step's list of pages to score (`prepare_step`), and their index scores, one persistent launch a layer (TMA page copies) |
| `attention/sparse_cute.py` | Eight CTAs a row: top-15 + local page, attention over them; the row's last CTA merges and writes o_proj's MXFP8 input |
| `attention/o_proj.py` | o_proj on FlashInfer's low-latency MXFP8 GEMM, one tactic per 8-, 16- or 32-row tile |
| `attention/ptx.py` | NVFP4 decode, warp MMAs, loads and stores in PTX |
| `moe/__init__.py` | `build_low_batch_ffn`, `LowBatchFfn` (`accepts`, `run`), `unsupported_reason`, `MAX_ROWS` |
| `moe/front_cute.py` | Routing + per-token NVFP4 input + routing record in one launch, one build per width (8, 16, 32 rows) |
| `moe/gemm_cute.py` | Routed FC1 (SwiGLU-OAI epilogue, FC2's input quantized per token tile) and FC2 on persistent `tcgen05` NVFP4 GEMMs |
| `moe/shared_cute.py` | The MXFP8 shared expert: input quantization, split-K FC1, SwiGLU-OAI + MXFP8, FC2 on the same GEMM, up to four 8-row tiles |

The routing record (`build_record` and its helpers, the ready words), the
persistent GEMM skeleton (`RoutedExpertGemm`: `gather_b_warp`,
`signal_ready`, `ready_target`) and the front's widths (`width`)
are imported from the DeepSeek-V4.1 low-batch MoE
(`deepseek_v41/nvidia/specialized_decode/moe/`); `test_kernels.py` builds every
kernel against them, so a change there that breaks these fails it.
`MiniMaxM3MoE.process_weights_after_loading` builds one `LowBatchFfn` per MoE
layer once the weights are processed; it compiles the kernels then, outside any
CUDA graph capture, and binds the routed and shared expert weights in place (no
copies). `MiniMaxM3SparseAttention.process_weights_after_loading` builds its
`LowBatchAttention` the same way (the kernels compile once per process) and
reads o_proj's weights in place.

## Paths

| Path | Rows |
| --- | --- |
| qkv projection, QK norm, RoPE and KV insert: stock | all |
| Index scores of every history page (E4M3 query, NVFP4 keys, max over the page) of the requests whose last query has more than 15 history pages, contexts up to 8192 pages (1M tokens) | 1-32, up to 32 requests of up to 8 queries |
| Per row the top-15 history pages (all of them when at most 15) and the local page, the causal attention over them (NVFP4 K/V, E4M3 query), the output quantized to MXFP8 as o_proj's input; padding requests (seq_len 0) write zeros | 1-32 |
| o_proj: low-latency MXFP8 GEMM, weight loads ahead of its PDL wait | 1-32 |
| Gate: stock `GateLinear` (FP32 logits) | all |
| Front: top-4 of `sigmoid(logit) + bias`, BF16 weights from the recovered scores, per-token NVFP4 of the input, routing record | 1-32 |
| Routed FC1 and FC2 on the record's work items; the last FC1 item of each token tile quantizes its BF16 output per token for FC2 | 1-32 |
| FC2 waits for the FC1 grid | 1-8 |
| FC2 starts on each token tile once FC1's items of it are counted (the record's ready words) | 9-32 |
| Shared expert on the aux stream, joined before `run` returns: MXFP8 input, FC1 split over K into 12 slices (144 work items per 8-row tile), BF16 gate_up and SwiGLU-OAI to MXFP8 `h`, FC2 | 1-32 |
| Top-k finalize, all-reduce, next input norm: the fused boundary kernel | up to 32 |

## Dispatch

`MiniMaxM3Model.forward` asks `SpecializedDecode.plan` for a `DecodeStep`;
it gets one only for 1-32 rows (`moe.MAX_ROWS`) without intermediate tensors
while the forward is being captured into a FULL CUDA graph
(`captures_pure_decode`), and then hands its layers to `DecodeStep.run`.
The step carries the attention metadata of the first sparse layer with a
`LowBatchAttention` (decode block table, seq_lens and query length, through
`attention.prepare_step`) when the step is all decode and its request and
query counts fit `accepts`; each sparse layer with a `LowBatchAttention` then
hands its q and index q to it after the KV insert, the others run their stock
attention. A MoE layer runs its `LowBatchFfn` when it defers its finalize for the
step's rows (`defers_finalize`) and `accepts(x)` (1-32 contiguous BF16 rows,
the bound weights); the other MoE layers run their stock modules. Every eager
forward, PIECEWISE capture, prefill and larger batch runs the stock layers.
The decisions read host state only, nothing is read back from the GPU, and
they are the same on every TP rank.
