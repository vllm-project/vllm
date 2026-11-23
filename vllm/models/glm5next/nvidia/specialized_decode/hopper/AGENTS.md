# GLM-5.3-Flash specialized decode on Hopper

`HopperSpecializedDecode` (`decode.py`) is the specialized decode executor on
H100/H200 (SM90), and `HopperSpecializedDraft` its MTP draft layer.
`SpecializedDecode.for_target` and `SpecializedDraft.for_target`
(`runtime/`) build them on SM90 instead of the GB300 executors. They keep the
GB300 executor's plan, layer loop and boundary order (`runtime/decode.py`,
`runtime/draft.py`) and replace its stages with the Hopper ones. With
`decode_backend` left at `auto` nothing here is imported.

## Scope

The deployment requirements are the parent package's (`runtime.unsupported`,
every unmet one listed at startup), except on SM90:

- KV cache: BF16 (`auto` with the BF16 model dtype, or `bfloat16`), the SM90
  sparse MLA recipe.
- Speculative decoding: MTP with at most 5 draft tokens (verify steps of up to
  6 tokens a request).

Only FULL CUDA graph captures of pure decode steps and uniform verify steps of
up to `MAX_TOKENS` (32) tokens run the executor: plain decode of 1-32 requests,
MTP5 verify steps of 1-5 requests (6/12/18/24 tokens at C1-C4). Every other
forward runs the stock layers, and so does a captured step that no Hopper stage
serves. Draft steps: draft prefills (q tokens a request) and draft decodes (one
token a request) up to 32 tokens.

## Stages

| Policy entry | Module | Class | Replaces |
| --- | --- | --- | --- |
| `kda` | `kda.py` | `HopperKdaLayer` | the KDA layer (in_proj, conv / delta rule / gated norm, o_proj) |
| `dsa`, `draft_dsa` | `dsa.py` | `HopperDsaLayer` | `Glm5NextMLAAttention.forward` (projections, indexer, kpool, KV insert, sparse MLA, W_UV, o_proj) of the backbone and of the draft |
| `moe`, `draft_moe` | `moe.py` | `HopperMoe` | `Glm5NextMoE.forward` (router, routed and shared experts) of the backbone and of the draft |
| `dense_mlp` | `dense.py` | `HopperDenseMlp` | the 3 dense layers' `Glm5NextMLP` |
| `boundary` | `boundary.py` | `HopperBoundary` | all-reduce + `hc_fused_post_pre` at every backbone boundary |
| `draft_boundary` | `boundary.py` | `HopperDraftAllReduceNorm` | the draft's all-reduce + residual add + RMSNorm, and its embedding all-reduce + enorm / hnorm |
| `logits_head`, `logits_gather` | `logits.py` | `HopperLogitsHead`, `HopperLogitsGather` | the LM head shard GEMM and the logits all-gather (eager) |

`steps.py` builds each stage from the stock layer once the weights are loaded.
Each stage returns its sublayer output TP-partial; `HopperLayerStages`
all-reduces it unless the boundary at that token count reduces it. Stock
sublayers leave their output TP-partial for the Hopper boundary (o_proj's
`reduce_results`, the MoE runner's `skip_final_all_reduce`, as on GB300).

## Fallbacks

Each stage runs stock, independently of the others, when:

- its policy entry is `""` or leaves the token count out
  (`additional_config["policy"]`, e.g. `{"moe": "1-8:auto"}`);
- its module is not part of the build (`steps.stage_class`: only a missing
  module falls back; an import error inside the module raises);
- it declines its layer: `of` returns None or the constructor raises
  `ValueError`, with the reason logged once (`steps.declined`); DSA layers on
  a sparse MLA backend other than `FLASHINFER_MLA_SPARSE_SM90` decline too;
- it does not support the step (`supports(tokens[, q])`).

With no stage built, `plan()` returns None for every step. Each capture logs a
line per rank: `Hopper specialized decode captured a <T>-token step (q=<q>):
boundary h/n, dense_mlp h/n, dsa h/n, kda h/n, moe h/n` (Hopper / all calls).

## Lifecycle

- Backbone: built by the model state after weight loading (MoE and dense
  stages compile in `prepare`); `prepare_kv` (every bound KV cache) compiles the
  KDA and DSA layers for the bound caches.
- Draft: built on the first eager draft forward (memory profiling, before the
  KV caches are bound); its DSA layers compile on the first eager forward after
  each binding (`HopperSpecializedDraft.plan`).
- Collectives: `HopperBoundary`, `HopperDraftAllReduceNorm` and
  `HopperLogitsGather` are built in lockstep on every TP rank. Every per-step
  choice reads host state that all ranks share (token counts, metadata host
  fields, the policy).

## Stock internals it relies on

- `Glm5NextDecoderLayer`: `layer_kind`, `layer_idx`, `_mlp_is_moe`,
  `hc_pre` / `hc_fused_post_pre` / `hc_post`, `hc_*_fn` / `scale` / `base`, the
  norms' `weight` and `variance_epsilon` (through `runtime/decode.py`).
- `self_attn.o_proj.reduce_results`, `mlp.experts.moe_config.skip_final_all_reduce`,
  `mlp.down_proj.reduce_results`: set for the call by `_override`.
- The draft: `Glm5NextMultiTokenPredictor._mtp_layers`, `prefill_rows`,
  `mtp_block.self_attn.mla_attn.mla_attn.kv_cache` (bound or the runner's
  empty placeholder) and `attn_backend`.

## Tests

- `tests/models/glm5next/hopper/test_runtime.py` (CPU): refusals, policy,
  dispatch, plan(), fallbacks, the layer loop on fake stages against the stock
  decoder and MTP forwards (two simulated TP ranks), and each stage module's
  interface as the executor calls it.
- `tests/models/glm5next/hopper/tp4_hopper.py` (4 SM90 GPUs, `GLM53_MODEL`):
  the served model with the switch on and off.
