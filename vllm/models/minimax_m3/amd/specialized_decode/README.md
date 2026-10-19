# MiniMax M3 specialized decode

`--monokernel` on MI355X/gfx950 runs each sparse layer (3-59) as vLLM's input
norm and attention followed by one persistent FlyDSL launch for the rest of the
layer (`mono/ffn.py`): the attention TP all-reduce and residual add, the Gemma
post-attention norm, the FP32 router and top-4, the shared and routed experts,
and the MoE TP all-reduce.

Milestone one is deliberately narrow:

- TP4 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU);
- eager plain decode, 1-16 rows; everything else runs the stock layers;
- BF16 expert activations against the `AITER_MXFP4_MXFP4` post-load weights
  (AITER `(16, 16)` weight shuffle, `e8m0_shuffle` scales), matching what
  native AITER computes at decode.

`runtime/` owns refusals, planning, health and shutdown; `mono/` owns the
kernel and its host launcher. Unsupported static deployments are refused with
all known reasons. In-kernel waits are bounded and set a sticky error word on
every rank; any failure, including a rank-local exception in a layer, stops
the step on every rank before logits are computed.
