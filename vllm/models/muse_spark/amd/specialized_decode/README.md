# Muse Spark specialized decode (MI300X / MI325X, FlyDSL)

`--monokernel` for `MuseSpark1xForCausalLM` (Quark PTPC FP8) on MI300X / MI325X:
decoder layers 1-60 of a plain decode step of 1-16 rows run as persistent
FlyDSL launches, one a layer, from the layer's attention output to the next
layer's QKV + o-gate. vLLM's QK-norm / RoPE / KV insert and attention op run
between the launches; layer 0, layer 61 and every step the kernels do not
cover run vLLM's stock path.

## Scope

- Deployment (checked at startup; a deployment that cannot run the kernels
  raises): gfx942 with 304 CUs, tensor parallel size 8 with
  `--enable-expert-parallel` (linear placement, 32 experts a rank, no EPLB,
  no sequence-parallel MoE), no PP / DP / DCP / PCP, no speculative decoding
  or LoRA, bf16. Dense layers: FP8 per-channel weights in the row-major
  layout (not AITER's preshuffled kernels), dynamic per-token inputs. Experts:
  row-major FP8 `w13` / `w2` with per-channel scales (Triton MoE).
- Steps (checked every forward; anything else runs stock): decode-only
  batches whose row count is a built width (the CUDA graph capture sizes up to
  `MAX_ROWS` = 16), FULL CUDA graph or eager, no aux hidden states. The kernels
  build and pass their test up to 128 rows; from 32 rows on stock decodes
  faster on MI325X, so wider steps run stock.

## A launch (layer i, 1 <= i <= 60)

All 304 CTAs; six grid barriers. The all-reduces are two-shot: every rank
scatters each group to its owner rank, the owner sums the eight copies in
rank order and pushes the bf16 sum to every rank.

     1  o-norm x sigmoid(o-gate) per 64-dim head, per-token FP8 quant
     2  o_proj (split-K 2), scatter          3  reduce + gather
     4  attention boundary (post-attn norm, gated residual, pre-FFN norm)
     5  router (fp32 as three bf16 planes) and pooler partials
     6  top-8 of sigmoid + bias, route weights, pre-expert norm, FP8 quant
     7  expert gate-up, SiLU x up, per-token FP8 quant of the activation
     8  expert down
     9  post-expert norm x route weight, summed per token, scatter
    10  reduce + gather                     11  unpooler
    12  FFN boundary (post-FFN norm, gated residual, layer i+1's input norm),
        per-token FP8 quant
    13  layer i+1's QKV + o-gate GEMM

## Contract with vLLM

- Weights: read in place from vLLM's parameters (`o_proj`, `qkv_proj`, the
  pooler / unpooler row slices, the experts); the fp32 router rows
  `[32 r, 32 r + 32)` with the output multiplier folded in, as three bf16
  planes; the norm multipliers as f32 copies (`weight + gain_center`).
  `refresh_weights` re-takes them after a reload.
- Quant: the dense inputs follow AITER's per-token quant (max 240), the MoE
  input `scaled_fp8_quant` (max 224, scale floor `1 / (224 * 512)`) and the
  activation `silu_mul_token_quant`; codes are `x * rcp(scale)`.
- Sync: an IPC mailbox (`mono/peer.py`) shared by all widths; tags come from
  a device epoch bumped once a step, so CUDA graph replays stay correct. A
  launch that waits too long on a peer or the grid gives up and records it;
  `poll_health` raises on the next poll.
