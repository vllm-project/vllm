# Kimi-K3 specialized decode (MI355X, MI325X)

An opt-in decode path for Kimi-K3 at TP8 on MI355X (gfx950) and MI325X (gfx942),
following RFC #60904.
A pure spec-verify step of at most 16 rows (8 on MI325X) runs each MoE decoder
layer as persistent FlyDSL MonoKernels (256 CTAs, all resident), in "layer mode",
inside the stock layer loop. The TP all-reduces run in-kernel over peer memory.

- **Whole layers** (KDA): K1 (AttnRes, in_proj, f_b, spec conv, KDA recurrence,
  gated RMSNorm), then K2.
- **FFN layers** (MLA, DCP1 or DCP8): M1a (AttnRes, qkv_a and g_proj, q / kv
  RMSNorm, the fp8 KV write of this rank's tokens, q_b, MXFP4 W_UK, the fp8 query
  gathered to every rank's peer memory), AITER's MLA decode on this rank's KV
  shard, then K2, which runs M1b first (the LSE combine of every rank's partials,
  MXFP4 W_UV, the sigmoid gate). At DCP1 every rank holds the whole KV: M1a writes
  its 12 heads' query directly and M1b reads AITER's output as is, with no gather
  and no combine.
- **FFN layers M1 cannot serve** (gfx942, a non-fp8 KV cache,
  `VLLM_ROCM_USE_AITER_FP4BMM` unset, or MLA weights M1 does not read): vLLM's
  AttnRes and MLA attention ops write o_proj's input, then K2.
- **K2**: o_proj + all-reduce + MLP AttnRes, then the latent MoE (router,
  top-16, routed experts, shared experts, latent all-reduce + RMSNorm, up_proj,
  final all-reduce).

The kernels read vLLM's loaded weights, KDA states and MLA KV cache in place.
Nothing is copied or re-quantized. The routed experts are the MXFP4 checkpoint on
MI355X, and on MI325X the packed-int4 requant vLLM's own MoE uses there. The
kernels tile up to 64 rows (`MAX_TOKENS`), but at 32 and 64 rows they lose to
vLLM's decode, so a served step is capped at 16 (`MAX_ROWS`; 8 on MI325X).

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER_FP4BMM=1 vllm serve <Kimi-K3> -tp 8 \
  [--decode-context-parallel-size 8 --dcp-comm-backend a2a] --kv-cache-dtype fp8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, ...}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

On MI325X, add vLLM's int4 requant of the routed experts:
`--quantization-config '{"moe": {"weight": "int4_per_group_32"}}'`. vLLM's MLA
there serves DSpark with decode context parallel 1, a BF16 KV cache (no
`--kv-cache-dtype fp8`) and `"attention_backend": "TRITON_MLA"` for the draft, and
a full step is at most 8 rows: `max_num_seqs` 1, 2 or 4 with
`max_num_seqs x (num_speculative_tokens + 1) <= 8`.

There are no other options. `decode_backend=auto` (the default) runs vLLM's decode.

## Supported deployments

TP8 on MI355X with all 256 CUs, or on MI325X (304 CUs; 256 run the kernels) with
the int4 expert requant, decode context parallel 1 or 8, speculative decoding with
at most 15 speculative tokens (7 on MI325X), Model Runner V2, BF16 conv / FP32
recurrent KDA states. The kernels run on any gfx942 with 304 CUs, but Kimi-K3 at
TP8 holds about 200 GiB of weights a rank, more than an MI300X has, with or
without them. M1 assumes no KV cache interleave (it writes the slots vLLM maps),
but at DCP8 vLLM's AITER MLA backend serves DSpark at interleave 1 only.

M1 runs on gfx950 only. It also needs an fp8 KV cache and MLA's W_UK / W_UV as
AITER MXFP4 bmms (`VLLM_ROCM_USE_AITER_FP4BMM=1`). Without them the MLA layers run
vLLM's attention ops, then K2, and startup logs why. The KDA layers run K1 and K2
either way.

KV transfer support is limited to built-in MoRIIO READ and WRITE consumers.
Keep the producer on `decode_backend=auto` and use matching cache layouts.

If you request the specialized decode for anything else, startup fails with one
`ValueError` that lists every unsupported setting:

- **Config, in the model's `__init__`:** pipeline, data, expert or prefill
  context parallelism, EPLB, DBO, LoRA, unsupported KV connectors,
  routed-expert capture, sleep mode, CPU weight offload, weight updates,
  no speculative decoding,
  adaptive verification, dynamic speculative decoding, ReplaySSM / RecoverSSM,
  the V1 model runner, `STOCK_TORCH_COMPILE`, another model's shapes, another
  GPU, a partitioned GPU, an `ARCH` other than the GPU's, gfx942 without the
  int4 expert requant.
- **Loaded weights, when the model state builds the executor:** each kernel
  layer's weight shapes and dtypes and the MoE backend (`AITER_MXFP4_BF16`, K3
  SiTU; on gfx942 its packed-int4 experts and their layout). Each MLA layer's
  shapes, gate, DCP setup, fp8 cache and bmm weights decide M1 or vLLM's ops (not
  a refusal).
- **KV caches, at every bind:** the KDA state dtypes and layout, and with M1 the
  MLA cache's fp8 dense rows.

Every TP rank agrees on each verdict before anything collective runs.

## Steps

`plan()` runs once per step. It picks the kernels for a pure spec-verify step of
whole uniform requests whose width is one of the built widths: every multiple of
`num_speculative_tokens + 1` up to 16 rows (8 on MI325X). With M1 the MLA layers' metadata must
agree (pure decode, the same width, AITER's non-Gluon decode). Every other step
runs vLLM's layers: prefill, mixed, wider and PIECEWISE steps.

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

## gfx942

The kernels compile for the GPU FlyDSL targets (the `ARCH` environment variable
overrides it; startup refuses an `ARCH` other than the GPU's). gfx942 runs the
same launches, CTAs and mailboxes as gfx950; the gfx950 code is unchanged.

| | gfx950 (MI355X) | gfx942 (MI325X) |
| --- | --- | --- |
| routed experts | MXFP4 as loaded; MXFP4 latent and intermediate | packed int4 (vLLM's requant, aiter's a16wi4 layout); bf16 latent and intermediate |
| LDS | 160 KB: whole token rows | 64 KB: token rows a K window at a time, the latent and the intermediate in K phases |
| served rows | 16 | 8 (the int4 MoE stages' LDS) |
| MLA layers | M1, else vLLM's ops | vLLM's ops (M1 needs MXFP4 MFMAs) |
| MFMA | 16x16x32 bf16; permlane swaps | two 16x16x16 bf16; `shuffle_xor` |

## Layout

Everything lives in this folder. The model files only call in: `linear.py`
(the config check in `__init__`, `plan_specialized_decode` in `forward`,
`step.layer(i)` in the layer loop, `after_specialized_decode_step` in
`compute_logits`), `model.py` (`get_model_state_cls`) and `mla.py`
(`skip_o_proj`). Nothing outside the model changes: M1a writes the query
AITER's cprr decode pads to (e.g. 96 -> 128 heads), and `runtime/mla.py` hands
it to `forward_mqa` in place of the backend's padding copy.

```text
hooks.py         SpecializedDecodeMixin: the model-side hooks
model_state.py   KimiK3ModelState: builds the executor after load, binds KV caches
runtime/
    __init__.py  DecodeConfig, specialized_decode_config(), unsupported(),
                 m1_reasons()
    decode.py    SpecializedDecode: plan() -> Step, Step.layer(i), prepare_kv(),
                 after_step()
    guards.py    weight and KV cache refusals, the fail-stop watch
    layers.py    the layer kernels: vLLM tensors -> launches
    mla.py       the MLA layers' decode as vLLM ops: the step's context,
                 AITER's decode (the M1a / M1b reference)
    runner.py    per-rank launch state: scratch, step epoch, health, peers
    peer.py      peer memory and TP agreement (adapted from ROCm/ATOM, MIT)
mono/            kernel source
    layer.py     K2
    attention/   kda.py (K1), mla.py (M1a), mla_back.py (M1b), back.py
    stages/      moe.py, gemv.py
    common/      device ops, mailbox sync, MX / int4 helpers, plan / build keys,
                 the target arch (adapted from ROCm/ATOM atom/mono, MIT)
    abi.py       the error-word layout
```
