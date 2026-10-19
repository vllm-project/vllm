# MiniMax M3 specialized decode

`--monokernel` on MI355X/gfx950 runs each sparse layer (3-59) as vLLM's input
norm and attention followed by one persistent FlyDSL launch for the rest of the
layer (`mono/ffn.py`): the attention TP all-reduce and residual add, the Gemma
post-attention norm, the FP32 router and top-4, the shared and routed experts,
and the MoE TP all-reduce.

Milestone one is deliberately narrow:

- TP4 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU);
- plain decode of 1-16 rows, eager or replayed from a FULL CUDA graph (graph
  padding rows are inert); PIECEWISE steps, larger and mixed steps run the
  stock layers;
- BF16 expert activations against the `AITER_MXFP4_MXFP4` post-load weights
  (AITER `(16, 16)` weight shuffle, `e8m0_shuffle` scales), matching what
  native AITER computes at decode.

`runtime/` owns refusals, planning, health and shutdown; `mono/` owns the
kernel and its host launcher. Unsupported static deployments are refused with
all known reasons. In-kernel waits are bounded and set a sticky error word on
every rank. The word is copied with each step's sampled tokens, and the model
runner raises instead of returning a step's tokens once it is set.
