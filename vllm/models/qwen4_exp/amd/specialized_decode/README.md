# Qwen3.8-Flash-Next specialized decode

`--monokernel` on MI355X/gfx950 runs each decode step of
Qwen3.8-Flash-Next (`amd/Qwen3.8-Flash-Next-Quark-MXFP4`) at TP2 as persistent
FlyDSL launches, four a layer, in step mode (`runtime/decode.py`):

1. the attention hyper-connection front (`mono/front.py`): the grouped Gemma
   norm, the down + inject GEMM, hc_silu, the up GEMM and the gate-mix of
   `GatedResidual.mix`, from the previous launch's sums of squares;
2. the GDN block (`mono/blocks.py`, `build_gdn`): in_proj_qkvz (MXFP4) and
   in_proj_ba, the conv update and the gated delta rule on the cached states,
   RMSNormGated; on the QSA layers vLLM's QSA attention runs instead;
3. the o_proj block: out_proj (MXFP4) / o_proj (BF16, the output gate), the TP
   all-reduce and the attention HC combine into the multi-stream state;
4. the MLP front, then the MoE block: the router's softmax top-10, the MXFP4
   experts with the fused shared expert (expert 512), the TP all-reduce and
   the MLP HC combine.

The o_proj and MoE blocks fold the HC combine into their all-reduce epilogue
and hand the next front the state's sums of squares, so the state is written
once a block and the norm needs no extra pass. Embedding, the per-layer
embeddings, the final mix, the head and sampling run vLLM's modules.

Supported:

- TP2 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU), Model
  Runner V2;
- plain decode of 1-16 rows, eager or replayed from a FULL CUDA graph (graph
  padding rows are inert); PIECEWISE steps, prefill, mixed and wider steps run
  vLLM's forward. Past 16 rows the MoE block spills and the stock decode is
  faster;
- the checkpoint's shapes (`mono/layout.py`, `SHAPES`) and formats: online
  MXFP4 GDN projections in the layout of whichever AITER FP4 GEMM vLLM picked
  for the shape (ASM or Triton), quantizing their inputs as that GEMM does,
  `AITER_MXFP4_MXFP4` experts with
  `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, a BF16 conv state and an FP32
  recurrent state with `--no-enable-prefix-caching`.

## Refusals

Default-deny, in three stages, each reporting every reason in one error:

- config, in the model's `__init__` (`runtime/__init__.py`, `unsupported`):
  TP other than 2, PP, DP, EP, EPLB, DCP, PCP, DBO, HC/MoE sequence
  parallelism, LoRA, KV connectors, routed-expert capture, speculative
  decoding (MTP and EAGLE3 included), sleep mode, CPU weight offload, weight
  updates, GDN prefix caching, other cache dtypes, an unfused shared expert,
  the V1 runner, `STOCK_TORCH_COMPILE`, other model shapes, and a platform
  other than unpartitioned gfx950;
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
in_proj, router, MoE and GEMV stages shared by the blocks; `mono/common/`
the device primitives, the tagged mailbox with its bounded waits, and the build
keys. The mailbox, peer memory (`runtime/peer.py`), GEMV stage and `FailStop`
are copies of the Kimi-K3 decode's (#60683, by hyukjlee, after ROCm/ATOM),
trimmed to what these kernels use, so this package imports nothing of
Kimi-K3's.

## Toolchain

FlyDSL comes in through AITER (`docker/Dockerfile.rocm_base`); measured with
FlyDSL 0.3.4.1 (AITER v0.1.25). Startup does not check the FlyDSL version;
`tests/models/qwen4_exp/amd_specialized_decode/test_kernels.py` checks the
kernels against vLLM's ops.
