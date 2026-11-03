# MiniMax-M3 specialized decode (NVIDIA)

Kernels for small decode batches of MiniMax-M3.1-Flash on B200 and B300,
enabled with `--additional-config '{"decode_backend": "latency_specialized"}'`
(the default, `auto`, keeps the stock decode). With it off, the model runs
exactly as without this package. Its first step covers the FFN half of the MoE
layers; the attention, the all-reduce, the dense layers and the MTP draft keep
their stock paths.

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

## Structure

| Path | Contents |
| --- | --- |
| `__init__.py` | The setting, the startup checks, the per-call capture gate |
| `moe/__init__.py` | `build_low_batch_ffn`, `LowBatchFfn` (`accepts`, `run`) |
| `moe/front_cute.py` | Routing + per-token NVFP4 input + routing record in one launch; FC2's input quantization |
| `moe/gemm_cute.py` | Routed FC1 (SwiGLU-OAI epilogue) and FC2 on persistent `tcgen05` NVFP4 GEMMs |
| `moe/shared_cute.py` | The MXFP8 shared expert: input quantization, split-K FC1, SwiGLU-OAI + MXFP8, FC2 on the same GEMM |

The routing record and the persistent GEMM skeleton are imported from the
DeepSeek-V4.1 low-batch MoE (`deepseek_v41/nvidia/specialized_decode/moe/`).
`MiniMaxM3MoE.process_weights_after_loading` builds one `LowBatchFfn` per MoE
layer once the weights are processed; it compiles the kernels then, outside any
CUDA graph capture, and binds the routed and shared expert weights in place (no
copies).

## Paths

| Path | Rows |
| --- | --- |
| Gate: stock `GateLinear` (FP32 logits) | all |
| Front: top-4 of `sigmoid(logit) + bias`, BF16 weights from the recovered scores, per-token NVFP4 of the input, routing record | 1-8 |
| Routed FC1 and FC2 on the record's work items, FC2's input quantized per token from FC1's BF16 output | 1-8 |
| Shared expert on the aux stream, joined before `run` returns: MXFP8 input, FC1 split over K into 12 slices (144 work items), BF16 gate_up and SwiGLU-OAI to MXFP8 `h`, FC2 | 1-8 |
| Top-k finalize, all-reduce, next input norm: the fused boundary kernel | up to 32 |

## Dispatch

`MiniMaxM3MoE.forward_unfinalized` runs the low-batch FFN only when it holds
one (the layers above), `accepts(x)` (1-8 contiguous BF16 rows, the bound
weights) and the forward is being captured into a FULL CUDA graph
(`captures_pure_decode`); every eager forward, PIECEWISE capture, prefill and
larger batch runs the stock code. The
gate reads host state only, nothing is read back from the GPU, and it is the
same on every TP rank.
