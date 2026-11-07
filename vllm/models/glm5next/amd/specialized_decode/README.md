# GLM-5.3-Flash specialized decode

`--monokernel` on MI355X/gfx950 runs each MoE decoder layer 3-43 of a decode
step as one persistent FlyDSL launch (256 CTAs, one per CU), inside vLLM's
layer loop. There are two launch kinds (`mono/ffn/kernel.py`):

- **KDA layers** (30): attention-side mHC seam, in-projections, short conv,
  KDA gates and recurrence, gated norm, o_proj and its TP reduce;
- **sparse-attention layers** (11): attention-side mHC seam, fused q_a / kv_a,
  KV and kpool index-cache writes, the indexer query (FWHT-128 + FP8), pool
  scoring and top-2048 selection, split attention, o_proj and its TP reduce;

each followed by the FFN half: mHC seam, RMSNorm, router, top-8, shared
expert, MXFP4 experts and the MoE TP reduce. Layers 0-2 and 44 (whose mHC
post contracts the streams) run vLLM's code.

## Supported

- Checkpoint: `amd/GLM-5.3-Flash-Quark-MXFP4` (exact geometry checked), routed
  experts on the `AITER_MXFP4_MXFP4` MoE backend; attention projections FP8
  block-128 or bf16.
- TP 2, 4 or 8 on dedicated, unpartitioned GPUs; Model Runner V2.
- Pure decode of 1, 2, 4 or 8 rows (widths 2/4/8; 2/4 at TP2), eager or
  replayed from a FULL CUDA graph. Prefill, mixed and larger steps and
  PIECEWISE replays run vLLM's layers.
- KV cache bf16, e4m3 (`--kv-cache-dtype fp8`) or `mxfp4_mla`; KDA recurrent
  state fp32.

## Layout

`runtime/` owns refusals, planning, cache binding and health:

- `spec.py`: `DecodeConfig`, config refusals, `specialized_decode_config()`;
- `guards.py`: refusals of the loaded model;
- `weights.py`: views of vLLM's loaded weights and their checks (no copy);
- `layers.py`: packing, cache binding, eligibility and the launch per layer;
- `decode.py`: `SpecializedDecode` (`plan()`, `bind_caches()`, `after_step()`)
  and `Step.layer()`;
- `routing.py`: the rank-uniform per-step decision;
- `failstop.py`: the health check of the kernels' error words.

`mono/` owns the kernels and their host launchers. `../model_state.py` builds
the decode after load and binds it to every KV cache vLLM allocates.

## Refusals

Unsupported deployments are refused with every reason at once: the config
before load (platform, MRv2, PP, DP, EP, EPLB, DCP, PCP, LoRA, KV connectors,
routed-expert capture, speculative decoding, sleep mode, CPU weight offload,
weight updates, prompt embeds, graph modes and capture sizes, geometry, TP
size, sequence-parallel MoE, cache dtypes), the model after load (MoE backend,
layer kinds, FlyDSL, compute units) and the caches at every bind (dtype, shape,
strides).

## Fail-stop

Every cross-CTA and cross-rank wait is bounded. An expired wait sets its
stage's sticky error word. After each step (`compute_logits`) the health check
reads the previous step's words without blocking the stream: the output rank
raises with the stage names and exits after a grace period, the other ranks
exit their workers. `VLLM_MONO_UNSAFE_WARN=1` logs and clears instead (debug
only); `VLLM_MONO_FAULT_INJECT=rank=R,step=N,us=D` stalls a rank for the fault
test.
