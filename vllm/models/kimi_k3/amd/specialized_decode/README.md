# Kimi-K3 specialized decode (MI355X)

An opt-in decode path for Kimi-K3 at TP8 on MI355X (gfx950), following RFC #60904.
A pure spec-verify step of at most 16 rows runs each MoE decoder layer as persistent
FlyDSL MonoKernels (256 CTAs, all resident), in "layer mode", inside the stock
layer loop. The TP all-reduces run in-kernel over peer memory.

- **Whole layers** (KDA): K1 (AttnRes, in_proj, f_b, spec conv, KDA recurrence,
  gated RMSNorm), then K2.
- **FFN layers** (MLA, DCP1 or DCP8): M1a (AttnRes, qkv_a and g_proj, q / kv
  RMSNorm, the fp8 KV write of this rank's tokens, q_b, MXFP4 W_UK, the fp8 query
  gathered to every rank's peer memory), AITER's MLA decode on this rank's KV
  shard, then K2, which runs M1b first (the LSE combine of every rank's partials,
  MXFP4 W_UV, the sigmoid gate). At DCP1 every rank holds the whole KV and an MLA
  layer is two launches: M1a writes its 12 heads' bf16 query, then K2 runs the
  decode attention itself (M1c, FlyDSL) and M1b merges M1c's KV splits. No gather
  and no AITER decode.
- **FFN layers M1 cannot serve** (DCP8 only: another MLA backend, a KV cache that
  is not fp8 E4M3, `VLLM_ROCM_USE_AITER_FP4BMM` unset, or MLA weights M1 does not
  read): vLLM's AttnRes and MLA attention ops write o_proj's input, then K2.
- **K2**: o_proj + all-reduce + MLP AttnRes, then the latent MoE (router,
  top-16, MXFP4 experts, shared experts, latent all-reduce + RMSNorm, up_proj,
  final all-reduce).

The kernels read vLLM's loaded weights, KDA states and MLA KV cache in place.
Nothing is copied or re-quantized. The kernels tile up to 64 rows (`MAX_TOKENS`),
but at 32 and 64 rows they lose to vLLM's decode, so a served step is capped at 16
(`MAX_ROWS`).

Kimi-K3.1 DSA layers run vLLM's prologue and indexer, then K2 runs the sparse
MLA itself (`VLLM_KIMI_SD_DSA`, default `auto`): split-K attention over the
indexer's top-k (`VLLM_KIMI_SD_DSA_SPLITS` slices per row, 16 by default) on
K2's top CTAs, with the logical-to-physical index conversion folded in
(`VLLM_KIMI_SD_DSA_FOLD`), then the split combine, the bf16 W_UV and the gate,
then o_proj and the latent MoE. `split` runs the attention as its own launch
before K2, `off` keeps vLLM's sparse MLA (K2 then takes the gated attention
output). `auto` runs `fused` at DCP1 with the `fp8_ds_mla` cache, a top-k of 2048,
a bf16 W_UV (no AITER FP4 / FP8 MLA bmms) and one request a step, else falls
back to `off` and logs why. DSA layers never
run M1. Eligible KDA layers retain K1/K2. Prefill and unsupported decode batches
use the standard layer path.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4BMM=1 \
  vllm serve <Kimi-K3> -tp 8 --max-num-seqs 1 \
  [--decode-context-parallel-size 8 --dcp-comm-backend a2a] --kv-cache-dtype fp8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, ...}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

There are no other required options. `decode_backend=auto` (the default) runs
vLLM's decode. On Kimi-K3.1 the fused DSA attention (`VLLM_KIMI_SD_DSA=auto`)
needs a bf16 W_UV, so leave `VLLM_ROCM_USE_AITER_FP4BMM` unset (or 0) there;
with it set, the DSA layers run vLLM's sparse MLA, then K2.

`latency_specialized` also turns on the c1 / c4 levers in `levers/`, each an env
flag that defaults to 1 and is ignored without it: `VLLM_K31_BEAT_MARKOV`,
`VLLM_K31_BEAT_DRAFTMETA`, `VLLM_K31_BEAT_K2SYNC`, `VLLM_K31_BEAT_EAGERPREP`,
`VLLM_K31_BEAT_KDAK1`, `VLLM_K3_INDEXER_SAMPLED_TOPK` and
`VLLM_KIMI_SD_DSA_PROLOGUE_INDEXER` (the last only with `VLLM_KIMI_SD_DSA_PROLOGUE=1`,
which defaults to 0, and `VLLM_K3_FUSED_INDEXER_PREP=1`). Set one to 0 to turn
that lever off.

## Drafters

The specialized decode serves the target model's layers. The draft runs vLLM's
DSpark speculator either way, with its own KV cache group(s). Both Kimi-K3.1
DSpark drafts are served:

- **Preview** (`Inferact/Kimi-K3.1-Pro-DSpark-preview`, `K3DSparkForCausalLM`):
  5 dense MLA layers (kv_lora 512) with their own fp8 cache on `ROCM_AITER_MLA`.
- **Release** (`Inferact/Kimi-K3.1-Pro-DSpark`, a speculators checkpoint served
  as `Qwen3DSparkForCausalLM`): 5 GQA layers (96 query / 16 KV heads, head dim
  64, 12 / 2 a TP rank) with a 2048-token sliding window, their own bf16
  sliding-window cache on `KIMI_K3_FLYDSL_SWA_GQA` (below), and a confidence
  head (read only by adaptive verification, which is refused).

`configure_dense_draft` (`kimi_k3/nvidia/dense_draft.py`) picks the draft's
backend and cache dtype unless `--speculative-config` sets
`attention_backend` / `kv_cache_dtype`. On ROCm a GQA draft gets
`KIMI_K3_FLYDSL_SWA_GQA` (`kimi_k3/amd/flydsl_swa_gqa_attn.py`): `TRITON_ATTN`
whose forward runs a FlyDSL split-K kernel (`ops/flydsl_swa_gqa*.py`, gfx950,
up to 16 query rows a request) for the draft's decode/verify calls; any other
call, or a layer it cannot serve, falls back to Triton and logs why.
`VLLM_K31_DRAFT_FLYDSL_ATTN=0` keeps plain `TRITON_ATTN`. `FLASH_ATTN` (the
default elsewhere) needs vllm_flash_attn, and `ROCM_ATTN`, `ROCM_AITER_FA` and
`ROCM_AITER_UNIFIED_ATTN` exclude the block-outermost (BLHNC) layout the DSA
indexer pins the worker to. Do not pick the draft's backend with the global
`--attention-backend`: that applies to the target's DSA layers too.

The levers by drafter:

- `VLLM_K31_BEAT_MARKOV`: both drafts (their `DSparkMarkovHead`, the target
  vocabulary, no draft-to-target map), greedy or probabilistic drafts, one
  request a rank (`ops/markov_sample.MAX_ROWS`). A confidence head the
  speculator uses is computed from the drafted tokens' Markov embeddings, as in
  vLLM's loop. Other drafts keep vLLM's loop and startup logs why.
- `VLLM_K31_BEAT_DRAFTMETA`: the target step skips the draft-only attention
  groups for both. The bs1 MLA planner replaces AITER's MLA schedule, which only
  the preview's MLA draft builds. The release's GQA backend (Triton or FlyDSL)
  plans none, so the planner is not used. Startup logs the draft's backend and
  which applies.
- `VLLM_K31_BEAT_K2SYNC`, `_EAGERPREP`, `_KDAK1`, `VLLM_K3_INDEXER_SAMPLED_TOPK`
  and the DSA prologue indexer touch only the target.

The release draft's sliding-window cache group turns off fine-grained
(`--prefix-match-unit`) prefix-cache hits: vLLM's sliding-window manager needs
block-aligned lookups.

## Supported deployments

TP8 on MI355X with all 256 CUs, decode context parallel 1 or 8, speculative
decoding, Model Runner V2, AITER (`VLLM_ROCM_USE_AITER=1`: K2's MoE is AITER's
SiTU MXFP4 MoE), BF16 conv / FP32 recurrent KDA states. `--max-num-seqs` is 1, 2, 4 or 8, with
max_num_seqs x (num_speculative_tokens + 1) <= 16 (a full step is one launch).
DCP1 runs M1c only: 3 or 7 speculative tokens, max_num_seqs 1 or 2. M1 assumes
no KV cache interleave (it writes the slots vLLM maps), but at DCP8 vLLM's AITER
MLA backend serves DSpark at interleave 1 only. A DSA model (Kimi-K3.1) runs no
M1, so none of the DCP1 limits apply to it; at any other max_num_seqs it serves
steps up to 8 rows (`FALLBACK_ROWS`) and runs wider steps with vLLM's decode.

M1 also needs the `ROCM_AITER_MLA` attention backend (it reads AITER MLA's decode
metadata), an fp8 E4M3 KV cache (`fp8` or `fp8_e4m3`) and MLA's W_UK / W_UV as
AITER MXFP4 bmms (`VLLM_ROCM_USE_AITER_FP4BMM=1`). Without them the MLA layers
run vLLM's attention ops, then K2, and startup logs why; DCP1 runs only with M1
and is refused. The KDA layers run K1 and K2 either way.

KV transfer support is limited to built-in MoRIIO READ and WRITE consumers and
the built-in Mooncake consumer.
Keep the producer on `decode_backend=auto` and use matching cache layouts.

If you request the specialized decode for anything else, startup fails with one
`ValueError` that lists every unsupported setting:

- **Config, in the model's `__init__`:** pipeline, data, expert or prefill
  context parallelism, EPLB, DBO, LoRA, AITER off, unsupported KV connectors,
  routed-expert capture, sleep mode, CPU weight offload, weight updates,
  no speculative decoding, an unserved max_num_seqs / speculative tokens / DCP,
  adaptive verification, dynamic speculative decoding, ReplaySSM / RecoverSSM,
  the V1 model runner, `STOCK_TORCH_COMPILE`, another model's shapes, another GPU.
- **Loaded weights, when the model state builds the executor:** each kernel
  layer's weight shapes and dtypes and the MoE backend (`AITER_MXFP4_BF16`, K3
  SiTU). Each MLA layer's attention backend, shapes, gate, DCP setup, fp8 E4M3
  cache and bmm weights decide M1 or vLLM's ops (not a refusal at DCP8).
- **KV caches, at every bind:** the KDA state dtypes and layout, and with M1 the
  MLA cache's fp8 dense rows.

Every TP rank agrees on each verdict before anything collective runs.

## Steps

`plan()` runs once per step. It picks the kernels for a pure spec-verify step of
whole uniform requests whose width is one of the built widths: every multiple of
`num_speculative_tokens + 1` up to `max_num_seqs` requests (at most 16 rows).
With M1 the MLA layers' metadata must agree (pure decode, the same width, AITER's
non-Gluon decode). Every other step runs vLLM's layers: prefill, mixed, wider and
PIECEWISE steps.

Under FULL graphs the plan is baked in at capture. vLLM's capture sizes are not
changed. A replay with fewer requests than its graph pads whole requests. Those
requests get the reserved null state slot and one accepted token, and the kernels
serve them like real ones. To capture every width exactly, pass them in
`--cudagraph-capture-sizes`.

Every launch's mailbox tag is the step epoch and a slot: K1 and M1a use the layer
index, K2 (and the M1b it runs) 128 + the layer index. The peer buffer holds K2's
regions, then M1a's gathered query, then M1b's partials (both M1 at DCP8 only). A
region is reused by the next layer only after K2's all-reduce, which every rank
reaches past its M1b.

## Fail-stop

Every in-kernel wait gives up after 10 s. The rank records the wait in its health
words (sticky: its later waits exit at once) and stops waiting, so a stalled peer
fails every rank instead of hanging them.

Each step, `compute_logits` queues a non-blocking copy of the rank's `FAILED`
word. Every 16 steps the TP ranks agree on what has landed over the TP CPU group,
and then every rank raises at the same step, so the engine stops. They must stop
together: a rank that raised alone would leave the others waiting in the next
collective (the drafter's) for ever.

Detection takes up to about 16 steps, so tokens of the failed steps can reach the
client before the engine stops. A zero-delay check needs a core hook and is left to
a follow-up.

`tests/models/kimi_k3/amd_specialized_decode/test_engine_fault.py` stalls a rank in
a served model and checks that the engine stops.

## Layout

Everything lives in this folder. The model files only call in: `linear.py`
(the config check in `__init__`, `plan_specialized_decode` in `forward`,
`step.layer(i)` in the layer loop, `after_specialized_decode_step` in
`compute_logits`), `model.py` (`get_model_state_cls`) and `mla.py`
(`skip_o_proj`). Nothing outside the model changes: M1a writes the query
AITER's cprr decode pads to (e.g. 96 -> 128 heads), and `runtime/mla.py` hands
it to `forward_mqa` in place of the backend's padding copy. The levers that
touch vLLM's v1 engine (the DSpark speculator, the AITER MLA builders, the GPU
model runner's sample) patch it at runtime from `levers/`, only once
`latency_specialized` is set.

```text
hooks.py         SpecializedDecodeMixin: the model-side hooks
model_state.py   KimiK3ModelState: builds the executor after load, binds KV caches
levers/          the c1 / c4 levers: the gate (lever_on), the v1 patches
    dspark.py    DSpark fused Markov sampling, the draft layers for DRAFTMETA,
                 the draft's attention backends (MLA or not) at startup
    mla.py       the bs1 MLA planner hook, the sparse MLA fused indptr
    mla_bs1_meta.py  the closed-form bs=1 MLA planner
    runner.py    the identity logits rows slice
runtime/
    __init__.py  DecodeConfig, specialized_decode_config(), unsupported(),
                 m1_reasons()
    decode.py    SpecializedDecode: plan() -> Step, Step.layer(i), prepare_kv(),
                 after_step()
    guards.py    weight and KV cache refusals, the fail-stop watch
    layers.py    the layer kernels: vLLM tensors -> launches
    mla.py       the MLA layers' decode as vLLM ops: the step's context,
                 AITER's decode (the M1a / M1b reference)
    dsa.py       the DSA layers: dsa_reasons(), DsaM1Layer (vLLM's prologue
                 and indexer, then K2 with the split-K sparse MLA)
    runner.py    per-rank launch state: scratch, step epoch, health, peers
    peer.py      peer memory and TP agreement (adapted from ROCm/ATOM, MIT)
mono/            kernel source
    layer.py     K2
    attention/   kda.py (K1), mla.py (M1a), mla_back.py (M1b),
                 mla_attn.py + mla_attn_geom.py (M1c), dsa_attn.py (DSA
                 split-K sparse MLA) + dsa_fold.py (its top-k fold's block
                 division), back.py
    stages/      moe.py, gemv.py
    common/      device ops, mailbox sync, MX helpers, plan / build keys
                 (adapted from ROCm/ATOM atom/mono, MIT)
    abi.py       the error-word layout
```
