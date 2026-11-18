# DeepSeek-V4-Pro mono decode layer (gfx950, FlyDSL)

One fused decoder layer for vLLM's decode steps on MI355X: the attention seam,
the attention, the attention all-reduce, the FFN seam, the MoE and the MoE
all-reduce, in two persistent launches a layer. This implementation builds on
the DeepSeek-V4.1-Flash mono decode layer (`models/deepseek_v41/amd/mono`).
[ROCm/ATOM's V4.1 mono decode](https://github.com/ROCm/ATOM/tree/97359d6df46c59736468168b7be8bc50b482a1a9/atom/models/deepseek_v41/mono)
is the reference design: its persistent execution framework, synchronization,
peer memory, mHC seam and MoE stages underpin this implementation. The adapted
files name their ATOM origins and retain attribution to Advanced Micro
Devices, Inc., under the MIT License. Here they are specialized to Pro's
widths and numerics; the attention stages (`attention/`) read vLLM's KV records
and weights as loaded.

## Scope

- Layers: the whole layer on all 61 backbone layers (CSA, compress ratio 4;
  HCA, ratio 128; layers 0-2 hash-routed). K1 takes the residual settled,
  without folding the previous output in, on layer 0 (the embedding broadcast
  to the four streams). Between K1 and K2 every layer hands vLLM its normed
  input and bf16 qr (`select`): vLLM's compressor runs there, and on a CSA
  layer its indexer; K2 then builds the key table, the CSA top-k through the
  block table or the HCA layer's every compressed row from vLLM's c128a table
  (`global_keys`), as wide as the longest context (8192 at 1M): its rounds,
  split chunks and combine cover a token's own keys only (`bounded`). The MTP /
  DSpark draft layers keep vLLM's path; the paths interleave freely (same
  tensors at the layer boundary).
- Steps: decode only (FULL graph or eager), M <= 64 rows (8 requests x 7
  DSpark tokens and spare), TP8, `fp8_ds_mla` KV records (ROCm).
- Contract: a drop-in for `DeepseekV4DecoderLayer.forward`:
  in  `x` (the previous FFN output, reduced, bf16 [M, 7168]), `residual`
      (bf16 [M, 4, 7168]), `post_mix` [M, 4, 1], `res_mix` [M, 4, 4] (f32)
  out the MoE output (reduced), the residual after the FFN seam, and that
      seam's post / comb / pre mixes.

## Launches

    K1  seam (stages/seam.py): slice (224 x 32 columns): fold x into the
        residual (post x + sum comb r), partial mixes (fn . R') and sums of
        squares -> gate (S): 24 mixes -> pre / post / comb (Sinkhorn 20) ->
        norm (S): the collapse (pre . R'), attn_norm -> vLLM MXFP8 -> X8
        attention front (attention/front.py): wqkv, q / kv norms + KV insert,
        wq_b + RoPE -> Q; the normed input and qr out for vLLM's compressor
        and indexer
    K2  key table (attention/front.py, stage_kt), from the top-k the indexer
        selected after K1
        attention back (attention/back.py): split attention (each head's q
        RMSNorm scaling its scores), combine + inverse RoPE, wo_a, wo_b; the
        bf16 wo_b partial pushed to every rank, or to its columns' owner
        FFN seam: reduce (the ranks' partials summed in rank order, fp32 ->
        bf16: the attention output), slice, gate, norm -> the MoE input
        MoE (stages/moe.py): xq, router (bf16 GEMV, f32 logits), route
        (sqrtsoftplus + bias, top-6, renormalized x 2.5; a hash layer's experts
        given), shared expert (MXFP8 GEMVs, clamped SiLU), ug (MXFP4 x MXFP8 per
        expert of the step's union), down (+ routing weight, top-k order;
        + shared) -> push, rank-order sum -> out

## Execution model and hand-offs

- Grid = 256 CTAs (one a CU, all resident), 512 threads a CTA, LDS a per-stage
  union. A stage is a task table: CTA `b` runs tasks `b, b + 256, ...`, stages
  in order; a consumer only waits on earlier stages, so no deadlock while the
  grid is resident. A GEMV task issues its weight loads before it polls its
  activations.
- In a launch: tagged pairs (value + tag in one 8 B store) or plain data +
  flags, at device scope. The tag is the launch pair's epoch (K1 `2e + 1`,
  K2 `2e + 2`), a device word K2's CTA 0 moves on once every CTA has marked
  the launch done: no clearing between steps, no host step hook, every
  argument a fixed device pointer (graph-safe). A step width and key-table
  width have a scratch of their own: under another layout a pair's tag word
  could be plain data.
- Across ranks: one symmetric uncached buffer a rank (aiter `UncachedIpcHeap`),
  every rank holding every peer's address; pushes are system-scope stores of
  (value, tag) pairs. Every rank runs the same launches, so the epochs agree;
  peer regions alternate by epoch parity, and a rank cannot run two steps ahead
  of a peer (each step's reduce waits on every rank's push).

## Numerics

- Seams: the gate is aiter's `mhc_fused_post_pre_delayed_rmsnorm` reduce (post
  fold order, fn as bf16 hi + lo, products summed in order, Triton sigmoid /
  exp / rcp), the collapse aiter's `mhc_pre` (the four streams in fp32 ->
  bf16, rsq(sum / H + eps), (x w) rstd). The gate matches the fp32 reference;
  aiter's fused seam is ~2e-3 off it.
- Attention: activations as MXFP8 (one E8M0 per 32 values, `code =
  clamp(ceil(log2(amax / 448)) + 127, 0, 254)`), where vLLM's path quantizes
  per 128 with fp32 scales: the kernels match the exact result no worse.
  Weights as vLLM loads them: wqkv, wo_b and the shared gate_up in aiter's
  (16, 16) shuffle, the others e4m3 [N, K] row-major with rows padded as
  `VLLM_ROCM_FP8_PADDING` pads them; the checkpoint's 128 x 128 E8M0 block
  scales repeated as [N/32, K/32] (exact). KV records as vLLM's insert writes
  them (dims 0..447 e4m3 with a UE8M0 per 64, RoPE dims bf16, 584 B); softmax
  in base 2 with the attention sink folded in.
- MoE: vLLM's router (bf16 gate GEMV, f32 logits, `topk_hash_softplus_sqrt`)
  and AITER a8w4 `fused_moe` (interleaved gate / up, swiglu limit 10, weights
  in aiter's (16, 16) shuffle and `shuffle_scale` order); the shared expert as
  MXFP8 linears with `silu_and_mul_with_clamp`.
- All-reduces (`stages/allreduce.py`): exact, fp32 in rank order -> bf16. One
  hop (every rank pushes its partial to every rank) or, from 42 rows, two (to
  the column's owner, which sums and pushes the bf16 sum to the others: 2 / tp
  of the bytes); bit-identical results.

## Validation

- Per layer against vLLM's own modules in a TP8 process group
  (`tests/models/test_dsv4_mono_numerics.py`), the protocol under graph
  replays; through the real model on random weights
  (`tests/models/test_dsv4_mono_layers.py`); the docking's path choice
  (`tests/models/test_dsv4_mono_decode.py`).
- Bench: layers of one type back to back in one HIP graph, cold weights, us
  per layer; then e2e (GSM8K, GPQA-Diamond, AIME25, RULER NIAH, AgentX as
  InferenceX runs it).
