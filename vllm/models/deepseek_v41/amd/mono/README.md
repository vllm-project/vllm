# DeepSeek-V4.1-Flash mono decode layer (gfx950, FlyDSL)

One fused decoder layer for vLLM's decode steps on MI355X: the attention seam,
the attention, the attention all-reduce, the FFN seam, the MoE and the MoE
all-reduce, in two persistent launches a layer. ATOM's V4.1 mono decode
(ROCm/ATOM `atom/models/deepseek_v41/mono`) is the reference design: the seam
and MoE stages (`stages/`) and the framework under them (`common/`) are adapted
from it, each file naming its origin; the attention stages (`attention/`) are
new, reading vLLM's KV records and weights as loaded.

## Scope

- Layers: every backbone layer with the standard seam and no compressor /
  indexer: 3-7, 9-13, 15-19 (compress ratio 2) and 21-23, 25-27, 29-31, 33-35,
  37-39 (ratio 1), 30 of 40. Layer 0 (embedding-broadcast seam), layer 1
  (Engram at the seam) and the compressor / indexer layers 2, 8, 14, 20, 24,
  28, 32, 36 keep vLLM's path; the two paths interleave freely (same tensors at
  the layer boundary).
- Steps: decode only (FULL graph or eager), M <= 48 rows (8 requests x 6
  DSpark tokens), TP2 / TP4, `fp8_ds_mla` KV records (ROCm), text tokens (the
  routing bias is `e_score_correction_bias`).
- Contract: a drop-in for `DeepseekV4DecoderLayer.forward`:
  in  `x` (the previous FFN output, reduced, bf16 [M, 5120]), `residual`
      (bf16 [M, 4, 5120]), `post_mix` [M, 4, 1], `res_mix` [M, 4, 4],
      `pre_mix` [M, 4] (f32, the previous seam's)
  out the MoE output (reduced), the residual after the FFN seam, and that
      seam's post / comb / pre mixes.

## Launches

    K1  seam (stages/seam.py): slice (160 x 32 columns): fold x into the
        residual (post x + sum comb r), the collapsed input (pre . R'), partial
        mixes (fn . R') and sums of squares -> gate (S): 24 mixes -> pre / post
        / comb (Sinkhorn 20) -> norm (S): attn_norm -> vLLM MXFP8 -> X8
        attention front (attention/front.py): wqkv, q / kv norms + KV insert,
        wq_b + RoPE -> Q
    K2  attention back (attention/back.py): split attention, combine + inverse
        RoPE, wo_a, wo_b; the bf16 wo_b partial pushed to every rank
        FFN seam: reduce (the ranks' partials summed in rank order, fp32 ->
        bf16: the attention output), slice, gate, norm -> the MoE input
        MoE (stages/moe.py): xq, router (bf16 GEMV, f32 logits), route
        (sqrtsoftplus + bias, top-6, renormalized x 1.5), shared expert (MXFP8
        GEMVs, clamped SiLU), ug (MXFP4 x MXFP8 per expert of the step's
        union), down (+ routing weight, top-k order; + shared) -> push, rank-order
        sum -> out

## Execution model and hand-offs

- Grid = 256 CTAs (one a CU, all resident), 512 threads a CTA, LDS a per-stage
  union. A stage is a task table: CTA `b` runs tasks `b, b + 256, ...`, stages
  in order; a consumer only waits on earlier stages, so no deadlock while the
  grid is resident. A GEMV task issues its weight loads before it polls its
  activations.
- In a launch: tagged pairs (value + tag in one 8 B store) or plain data +
  flags, at device scope. The tag is the launch pair's epoch (K1 `2e + 1`,
  K2 `2e + 2`), a device word K2's CTA 0 moves on once every CTA has marked
  the launch done: no clearing between steps, no host step hook, every argument
  a fixed device pointer (graph-safe).
- Across ranks: one symmetric uncached buffer a rank (aiter `UncachedIpcHeap`),
  every rank holding every peer's address; pushes are system-scope stores of
  (value, tag) pairs. Every rank runs the same launches, so the epochs agree;
  peer regions alternate by epoch parity, and a rank cannot run two steps ahead
  of a peer (each step's reduce waits on every rank's push).

## Numerics (the vLLM path each stage reproduces)

- Seams: aiter `mhc_fused_post_pre_delayed_rmsnorm` (post fold order, fn as
  bf16 hi + lo, products summed in order, Triton sigmoid / exp / rcp).
- Attention: vLLM's ROCm chain. `mxfp8(x)`: one E8M0 per 32 values,
  `code = clamp(ceil(log2(amax / 448)) + 127, 0, 254)`; weights in vLLM's
  row-major e4m3 [N, K] with [N/32, K/32] E8M0 scales (one shared copy); KV
  records as vLLM's insert writes them (dims 0..447 e4m3 with a UE8M0 per 64,
  RoPE dims bf16, 584 B); keys = the sliding window plus the top-k compressed
  rows, softmax in base 2 with the attention sink folded in.
- MoE: vLLM's router (bf16 gate GEMV, f32 logits, `topk_hash_softplus_sqrt`)
  and AITER a8w4 `fused_moe` (interleaved gate / up, swiglu limit 10, weights
  in aiter's (16, 16) shuffle and `shuffle_scale` order); the shared expert as
  vLLM's MXFP8 block-32 linears with `silu_and_mul_with_clamp`.
- All-reduces: exact, fp32 in rank order -> bf16. vLLM's decode all-reduces at
  M <= 48 are below its INT4 quick-reduce threshold, so vLLM runs them exact too.

## Validation

- Per layer against vLLM's own modules in a TP process group (`DeepseekV4MoE`,
  the mHC seams, the attention chain, the all-reduce), real checkpoint weights.
- Bench: layers of one type back to back in one HIP graph, cold weights,
  us per layer, M = 1..48, TP2 / TP4; then e2e (AgentX, InferenceX#3690
  recipe).
