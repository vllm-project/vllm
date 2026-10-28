# Qwen3.8-2.4T specialized decode

`--monokernel` on MI355X/gfx950 runs each decode step of Qwen3.8-2.4T
(`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`) at TP8 as persistent FlyDSL launches,
two a layer (`runtime/decode.py`):

1. the GDN block (`mono/blocks.py`, `build_gdn`), on the linear attention
   layers: the residual add + input_layernorm, in_proj_qkvz (MXFP4, its input
   quantized from the normed rows in global memory) and in_proj_ba, the conv
   update and the gated delta rule on the cached states, RMSNormGated (SiLU
   gate); on the full attention layers vLLM's input norm and attention run
   instead, up to o_proj's input (the fused q / k norm, RoPE and gate split);
2. the layer block (`build_layer`): out_proj (MXFP4) / o_proj (BF16, the
   output gate), the TP all-reduce, the residual add +
   post_attention_layernorm, the router's softmax top-10 and the shared
   expert gate, the MXFP4 experts with the fused shared expert (expert 512;
   the intermediate quantized to MXFP4 where it is produced, as the stock
   kernel does) and the TP all-reduce.

The embedding, the final norm, the head and sampling run vLLM's modules.

Supported:

- TP8 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU), Model
  Runner V2;
- plain decode of 1-16 rows, eager or replayed from a FULL CUDA graph (graph
  padding rows are inert); PIECEWISE steps, prefill, mixed and wider steps run
  vLLM's forward;
- the checkpoint's shapes (`mono/layout.py`, `SHAPES`) and formats: online
  MXFP4 GDN projections in the layout of whichever AITER FP4 GEMM vLLM picked
  for the shape (ASM or Triton), quantizing their inputs as that GEMM does,
  `AITER_MXFP4_MXFP4` experts with
  `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, a BF16 conv state and an FP32
  recurrent state with `--no-enable-prefix-caching`. Any KV cache dtype vLLM's
  attention takes.

## Refusals

Default-deny, in three stages, each reporting every reason in one error:

- config, in the model's `__init__` (`runtime/__init__.py`, `unsupported`):
  an architecture other than `Qwen3_5MoeForCausalLM`, TP other than 8, PP,
  DP, EP, EPLB, DCP, PCP, DBO, sequence-parallel MoE, LoRA, KV connectors,
  routed-expert capture, speculative decoding, sleep mode, CPU weight offload,
  weight updates, GDN prefix caching, other GDN cache dtypes, an unfused
  shared expert, the V1 runner, `STOCK_TORCH_COMPILE`, other model shapes,
  unnormalized top-k weights, layer scale, and a platform other than
  unpartitioned gfx950;
- model, after load (`runtime/guards.py`, `model_reasons`): every layer's
  modules, quantization methods, weight layouts and shapes;
- caches, at every KV cache bind (`cache_reasons`): the GDN states' dtypes,
  shapes and slot layout.

The TP ranks agree on every stage, so a rank never refuses alone. Without
`--monokernel` nothing here runs.

## Steps

`plan()` runs once a forward: it returns a `Step` for a pure decode step of
1-16 rows whose GDN metadata (identical on every rank) has int32 state indices
covering the rows, and None otherwise (vLLM's forward runs). Every kernel is
compiled when the executor is built (FlyDSL's `COMPILE_ONLY`), not on the
first step of a width.

## Failure

A fault in a launch traps the GPU queue: the worker's wait for the step's
output fails and the engine stops, no token is returned from that step.

Every hand-off between CTAs and TP ranks is a bounded wait (`mono/abi.py`):
a wait that gives up after 10 s records itself in the rank's health words,
raises the rank's flag on the output rank and marks the rank failed, so its
later waits exit at once and the step runs to its end. `FailStop`
(`runtime/guards.py`) reads the failed word after every step
without blocking and the TP ranks agree on it every 16 steps, then all raise
together and the engine stops.

## Code

`runtime/` owns the refusals, planning, the executor and the KV binding;
`mono/` owns the kernels and their launchers; `mono/stages/` holds the GDN,
norm, MoE and GEMV stages the blocks share; `mono/common/` the device
primitives, the tagged mailbox with its bounded waits, and the build keys.
The mailbox, peer memory (`runtime/peer.py`), GEMV stage and `FailStop` are
copies of the Kimi-K3 decode's (#60683, by hyukjlee, after ROCm/ATOM),
trimmed to what these kernels use, so this package imports nothing of
Kimi-K3's.

## Toolchain

FlyDSL comes in through AITER (`docker/Dockerfile.rocm_base`). Startup does
not check the FlyDSL version;
`tests/models/qwen3_5/amd_specialized_decode/test_kernels.py` checks the
kernels against vLLM's ops.
