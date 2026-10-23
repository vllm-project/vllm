# Muse Spark decode monokernel (MI355X, FlyDSL)

`--monokernel` for `MuseSpark1xForCausalLM` on MI355X: decoder layers 1-60 of
a speculative verify step run as persistent FlyDSL launches, from one layer's
attention output to the next layer's q, o-gate and KV insert. vLLM's own
attention op runs between the launches; layer 0, layer 61 and every step the
kernels do not cover run vLLM's stock path.

## Scope

- Deployment (checked at startup; a deployment that cannot run the kernels
  raises): gfx950 with 256 CUs, tensor parallel size 8 with
  `--enable-expert-parallel` (linear placement, 32 experts a rank, no EPLB),
  no PP / DP / DCP, Quark OCP-MXFP4 W4A4 experts on `AITER_MXFP4_MXFP4`
  (shuffled weights and scales), the fused QK-norm / RoPE / KV insert,
  `ROCM_AITER_FA` with a bf16 KV cache in the LBHNC layout (block size 16),
  DSpark with 7 draft tokens, no LoRA, no KV connector or offloading.
- Steps (checked every forward; anything else runs stock): uniform decode
  verify batches of 1-16 requests x 8 tokens, FULL CUDA graph or eager.
  8 / 16 rows run K3 (S = 8 / 16); 24-128 rows run HA + HM + HB with the rows
  padded up to S = 32 / 64 / 96 / 128 (`nrows` = the real rows).
- Every rank takes the same path: the decision uses only the batch shape and
  the attention metadata counts, which vLLM gives every rank alike.

## Launches (layer i, 1 <= i <= 60)

    K3  o-norm x gate, o_proj, all-reduce, attention boundary (post-attn norm,
        gated residual, pre-FFN norm), router + pre-expert projection, top-8,
        pre-expert norm, the local experts (gate-up, SwiGLU, down, post-expert
        norm, weighted sum), EP all-reduce, post-expert projection, FFN
        boundary (post-FFN norm, gated residual, layer i+1's input norm),
        layer i+1's QKV + O-gate GEMV, QK-norm, RoPE, KV insert
    HA  K3 up to the pre-expert norm; the residual stream token-sharded
    HM  the local experts; the partials pushed to their owner ranks
    HB  EP reduce, post-expert projection, FFN boundary, QKV, KV insert; the
        stream all-gathered only where a full copy is read (DSpark aux layers,
        layer 61)

## Contract with vLLM

- Weights: read in place from vLLM's parameters (`kernels/` builds take
  vLLM's layouts): QKV + O-gate rows `[q | k | v | o_gate]` per rank,
  `o_proj`, the fp32 router with the output multiplier folded in (rows
  `[32 r, 32 r + 32)`), `pooler` / `unpooler` row slices, the experts after
  `shuffle_weight(layout=(16, 16))` and the scales after `e8m0_shuffle`. The
  norm multipliers are small bf16 copies (`weight + gain_center`).
- KV: layer i+1's `kv_cache` `[block][kv head][16 tokens][K 64 | V 64]` bf16
  and its group's `slot_mapping` (`slot < 0` = graph padding, skipped); the
  block stride is a launch argument.
- q leaves without the QK scale: vLLM's attention applies
  `0.125 * qk_scale_factor` as its softmax scale.
- Peers: one `cudaMalloc` buffer a rank and build, opened by every rank through
  IPC handles exchanged on the TP CPU group (plain device memory, not the
  uncached custom all-reduce buffers). Buffers are allocated at load time and
  every launch argument is a fixed pointer, so the launches are captured in
  vLLM's FULL graphs.
- Health: a wait on a peer rank times out instead of hanging and leaves a
  record in the build's health words; `compute_logits` copies them back
  asynchronously every 64 steps and raises on a timeout.

## Numerics

The kernels keep the MoE activations in bf16 (MXFP4 weights x bf16
activations). The stock W4A4 path quantizes the expert inputs and the SwiGLU
output to MXFP4, so `--monokernel` outputs differ from the stock path (and
prefill, which runs stock, uses W4A4). Other differences are reduction order
(router, GEMVs, all-reduces in rank order) and the bf16 norm multipliers.
