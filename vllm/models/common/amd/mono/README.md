# Mono decode layer (gfx950, FlyDSL)

One fused decoder layer for vLLM's decode steps on MI355X: the attention seam,
the attention, the attention all-reduce, the FFN seam, the MoE and the MoE
all-reduce, in two persistent launches a layer. This package is the shared
implementation; a model docks onto it with a `MonoSpec` and a `MonoDecode`
subclass (see "Adding a model"):

| model | docking | TP | layers |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | `models/deepseek_v41/amd/mono_decode.py` | 2, 4 | all 40 (ratio 0 / 1 / 2, index sources, Engram) |
| DeepSeek-V4-Pro | `models/deepseek_v4/amd/mono_decode.py` | 8 | all 61 (CSA ratio 4, HCA ratio 128, 3 hash layers) |

A layer outside the whole-layer kernels keeps vLLM's attention and runs the
rest -- the attention all-reduce, the FFN seam, the MoE and its all-reduce --
as one launch (the FFN launch). ATOM's V4.1 mono decode
(ROCm/ATOM `atom/models/deepseek_v41/mono`) is the reference design: the seam
and MoE stages (`stages/`) and the framework under them (`common/`) are adapted
from it, each file naming its origin; the attention stages (`attention/`) are
new, reading vLLM's KV records and weights as loaded.

## Layout

    spec.py       MonoSpec (a model's layer: widths, numerics, variants) and
                  Dims (a TP rank's share and every count derived from it)
    common/       the execution model and build keys (plan), wave / math ops,
                  the tagged mailbox (sync), MX arithmetic, peer memory
    stages/       the mHC seams (seam), the FP8 GEMV (gemv), the MoE: router,
                  route (bias top-k or hash), shared expert, routed ug / down
                  (moe, moe_shape, moe_plan, router), the in-kernel all-reduce
                  of the attention and MoE outputs (allreduce)
    attention/    the DeepSeek-V4 family's sparse MLA: K1's front (wqkv, q / kv
                  norms, KV insert, wq_b), K2's back (split attention over the
                  key table, combine, wo_a, wo_b), their GEMV and plan
    layer.py      the launches: K1, K2 and the FFN launch, from a MonoBuild
                  (spec, step width, TP, the layer's variant)
    runner.py     the host side: buffers, peer memory, the kernels a build key,
                  the launches' arguments (MonoRunner, the weights' checks)
    decode.py     vLLM's side every model shares: the deployment, the runner
                  of a spec, a step's eligibility, the caches as records
    sources.py    the package's source digest, in every build's JIT cache key

## Scope

- Steps: decode only (FULL graph or eager), at most `spec.max_tokens` rows
  (V4.1: 48 = 8 requests x 6 DSpark tokens; V4-Pro: 64 >= 8 x 7), the TP sizes
  of the docking, `fp8_ds_mla` KV records (ROCm). The DSpark / MTP draft
  layers keep vLLM's path; the paths interleave freely (same tensors at the
  layer boundary).
- Contract: a drop-in for the decoder layer's forward: in the previous FFN
  output (reduced) and the residual streams with their owed post / comb mixes
  (V4.1 also the previous seam's pre-mix: its seams are delayed); out the MoE
  output (reduced), the residual after the FFN seam, and that seam's mixes.
- Between K1 and K2 a layer may hand vLLM its normed input and bf16 qr
  (`select`): V4.1's index sources run their compressor and indexer there;
  every V4-Pro layer runs its compressor (and a CSA layer its indexer). K2
  then builds the key table: the top-k through the block table, or (an HCA
  layer, `global_keys`) every compressed row from vLLM's c128a table, as wide
  as the longest context (8192 at 1M): its rounds, split chunks and combine
  cover a token's own keys only (`bounded`).

## Launches

    K1  seam (stages/seam.py): slice (hidden / 32 tasks): fold x into the
        residual (post x + sum comb r), partial mixes (fn . R') and sums of
        squares -> gate (S): 24 mixes -> pre / post / comb (Sinkhorn 20) ->
        norm (S): the collapse (pre . R': the slice's with the incoming pre on a
        delayed seam, the gate's own otherwise), attn_norm -> MXFP8 -> X8
        attention front (attention/front.py): wqkv, q / kv norms + KV insert,
        wq_b + RoPE -> Q
    K2  attention back (attention/back.py): split attention (a head's q norm
        scaling its scores, V4), combine + inverse RoPE, wo_a, wo_b; the bf16
        wo_b partial pushed to every rank
        FFN seam: reduce (the ranks' partials summed in rank order, fp32 ->
        bf16: the attention output), slice, gate, norm -> the MoE input
        MoE (stages/moe.py): xq, router (bf16 GEMV, f32 logits), route
        (sqrtsoftplus + bias, top-k, renormalized x the routed scale; a hash
        layer's experts given), shared expert (MXFP8 GEMVs, clamped SiLU), ug
        (MXFP4 x MXFP8 per expert of the step's union), down (+ routing
        weight, top-k order; + shared) -> push, rank-order sum -> out
    FFN K2 without the attention back: push of vLLM's unreduced wo_b output,
        as its bits, to every rank's attention region, then K2's FFN seam and
        MoE (the same code, `layer.run_ffn`)

## Execution model and hand-offs

- Grid = 256 CTAs (one a CU, all resident), 512 threads a CTA, LDS a per-stage
  union. A stage is a task table: CTA `b` runs tasks `b, b + 256, ...`, stages
  in order; a consumer only waits on earlier stages, so no deadlock while the
  grid is resident. A GEMV task issues its weight loads before it polls its
  activations.
- In a launch: tagged pairs (value + tag in one 8 B store) or plain data +
  flags, at device scope. The tag is the launch pair's epoch (K1 `2e + 1`,
  K2 or the FFN launch `2e + 2`), a device word their CTA 0 moves on once every
  CTA has marked the launch done: no clearing between steps, no host step
  hook, every argument a fixed device pointer (graph-safe). A step width and
  key-table width have a scratch of their own: under another layout a pair's
  tag word could be plain data.
- Across ranks: one symmetric uncached buffer a rank (aiter `UncachedIpcHeap`),
  every rank holding every peer's address; pushes are system-scope stores of
  (value, tag) pairs. Every rank runs the same launches, so the epochs agree;
  peer regions alternate by epoch parity, and a rank cannot run two steps ahead
  of a peer (each step's reduce waits on every rank's push).

## Numerics

- Seams: V4.1's are aiter's `mhc_fused_post_pre_delayed_rmsnorm` (post fold
  order, fn as bf16 hi + lo, products summed in order, Triton sigmoid / exp /
  rcp); V4-Pro's the same gate on its own residual, its collapse aiter's
  `mhc_pre` (the four streams in fp32 -> bf16, rsq(sum / H + eps), (x w) rstd).
  The gate matches the fp32 reference; aiter's fused seam is ~2e-3 off it.
- Attention: activations as MXFP8 (one E8M0 per 32 values, `code =
  clamp(ceil(log2(amax / 448)) + 127, 0, 254)`): V4.1's vLLM path exactly;
  V4-Pro's vLLM path quantizes per 128 with fp32 scales, which the kernels
  match no worse against the exact result. Weights as vLLM loads them: e4m3
  [N, K] row-major (V4-Pro: rows padded as `VLLM_ROCM_FP8_PADDING` pads them)
  or in aiter's (16, 16) shuffle (V4-Pro's wqkv / wo_b / shared gate_up), the
  E8M0 scales as [N/32, K/32] (V4-Pro's 128 x 128 blocks repeated, exact); KV
  records as vLLM's insert writes them (dims 0..447 e4m3 with a UE8M0 per 64,
  RoPE dims bf16, 584 B); softmax in base 2 with the attention sink folded in.
- MoE: vLLM's router (bf16 gate GEMV, f32 logits, `topk_hash_softplus_sqrt`)
  and AITER a8w4 `fused_moe` (interleaved gate / up, swiglu limit 10, weights
  in aiter's (16, 16) shuffle and `shuffle_scale` order); the shared expert as
  MXFP8 linears with `silu_and_mul_with_clamp`.
- All-reduces (`stages/allreduce.py`): exact, fp32 in rank order -> bf16. One
  hop (every rank pushes its partial to every rank) or, from
  `spec.two_hop_rows` rows, two (to the column's owner, which sums and pushes
  the bf16 sum to the others: 2 / tp of the bytes); bit-identical results.

## Adding a model

A model of the family -- mHC seams, the sparse MLA attention over fp8_ds_mla
records, a routed MXFP4 MoE with a shared FP8 expert -- needs no kernel code:

1. `MonoSpec` (`spec.py`): its widths (hidden, q rank, heads, output groups,
   top-k, experts, intermediates), numerics (routed scale, swiglu limit, norm
   eps), variants (`delayed_seam`, `q_head_norm`, `preshuffled`,
   `padded_rows`) and all-reduce (`two_hop_rows`). `Dims` derives every
   count; a width a stage cannot take fails at build time (a TP size the
   widths do not divide: `ShardError`).
2. A `MonoDecode` subclass next to the model (`decode.py`): `SPEC`,
   `TP_SIZES`, `create` (the layer checks: what the kernels serve, else None
   or the FFN launch), `weights` (the layer's tensors as `MonoLayerWeights`;
   `mx_scales` makes block scales the kernels' 32 x 32 ones) and the layer's
   forward hook (`runner.forward` / `run_ffn`, its `select` hook between K1
   and K2).
3. A numerics test like `tests/models/test_dsv4_mono_numerics.py`: vLLM's
   own op chain on random weights in the checkpoint's formats, each stage on
   its own input, and the exact fp32 result where the two quantize
   differently; then the layer test through the real model, gsm8k, AgentX.

A new variant of a stage is a flag on the spec or the build key read with
`const_expr`, so the other models' kernels stay as they are (the ISA of every
build of the others can be diffed: `FLYDSL_DUMP_IR`, `COMPILE_ONLY`). A model
outside the family (another attention, a dense FFN) reuses `common/`, the MoE
and the seams, and adds its own stages next to `attention/`.

## Validation

- Per layer against vLLM's own modules in a TP process group
  (`tests/models/test_dsv41_mono_numerics.py` at TP2 / TP4,
  `tests/models/test_dsv4_mono_numerics.py` at TP8), the protocol under
  graph replays; through the real model (`test_dsv41_mono_layers.py`).
- Bench: layers of one type back to back in one HIP graph, cold weights, us
  per layer; then e2e (gsm8k, AgentX as InferenceX runs it).
