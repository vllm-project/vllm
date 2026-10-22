# Kimi-K3 specialized decode (MI355X, MI325X)

An opt-in decode path for Kimi-K3 at TP8 on MI355X (gfx950) and MI325X (gfx942),
following RFC #60904.
A pure spec-verify step of at most 8 rows runs each MoE decoder layer as persistent
FlyDSL MonoKernels (256 CTAs, all resident), in "layer mode", inside the stock
layer loop. The TP all-reduces run in-kernel over peer memory.

- **Whole layers** (KDA): K1 (AttnRes, in_proj, f_b, spec conv, KDA recurrence,
  gated RMSNorm), then K2.
- **FFN layers** (MLA): vLLM's AttnRes and MLA up to o_proj's input, then K2.
- **K2**: o_proj + all-reduce + MLP AttnRes, then the latent MoE (router,
  top-16, routed experts, shared experts, latent all-reduce + RMSNorm, up_proj,
  final all-reduce).

The kernels read vLLM's loaded weights and KDA states in place. Nothing is copied
or re-quantized for them. The routed experts are the MXFP4 checkpoint on MI355X,
and on MI325X the packed-int4 requant vLLM's own MoE uses there.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <Kimi-K3> -tp 8 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, ...}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

On MI325X, add vLLM's int4 requant of the routed experts:
`--quantization-config '{"moe": {"weight": "int4_per_group_32"}}'`. vLLM's MLA
there serves DSpark with decode context parallel 1, a BF16 KV cache and
`"attention_backend": "TRITON_MLA"` for the draft.

There are no other options. `decode_backend=auto` (the default) runs vLLM's decode.

## Supported deployments

TP8 on MI355X with all 256 CUs, or on MI325X (304 CUs; 256 run the kernels) with
the int4 expert requant, decode context parallel 1 or 8, speculative decoding with
at most 7 speculative tokens, Model Runner V2, BF16 conv / FP32 recurrent KDA
states. The kernels run on any gfx942 with 304 CUs, but Kimi-K3 at TP8 holds
about 200 GiB of weights a rank, more than an MI300X has, with or without them.

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
  layer's weight shapes and dtypes, and the MoE backend (`AITER_MXFP4_BF16`, K3
  SiTU; on gfx942 its packed-int4 experts and their layout).
- **KV caches, at every bind:** the KDA state dtypes and layout.

Every TP rank agrees on each verdict before anything collective runs.

## Steps

`plan()` runs once per step. It picks the kernels for a pure spec-verify step of
whole uniform requests whose width is one of the built widths: every multiple of
`num_speculative_tokens + 1` up to 8 rows. Every other step runs vLLM's layers:
prefill, mixed, wider and PIECEWISE steps.

Under FULL graphs the plan is baked in at capture. vLLM's capture sizes are not
changed. A replay with fewer requests than its graph pads whole requests. Those
requests get the reserved null state slot and one accepted token, and the kernels
serve them like real ones. For example, at `num_speculative_tokens=1` three
requests replay the 4-request graph. To capture every width exactly, pass them in
`--cudagraph-capture-sizes`.

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
| MFMA | 16x16x32 bf16; permlane swaps | two 16x16x16 bf16; `shuffle_xor` |

## Layout

Everything lives in this folder. The model files only call in: `linear.py`
(the config check in `__init__`, `plan_specialized_decode` in `forward`,
`step.layer(i)` in the layer loop, `after_specialized_decode_step` in
`compute_logits`), `model.py` (`get_model_state_cls`) and `mla.py`
(`skip_o_proj`, o_proj's input for the MLA layers' K2).

```text
hooks.py         SpecializedDecodeMixin: the model-side hooks
model_state.py   KimiK3ModelState: builds the executor after load, binds KV caches
runtime/
    __init__.py  DecodeConfig, specialized_decode_config(), unsupported()
    decode.py    SpecializedDecode: plan() -> Step, Step.layer(i), prepare_kv(),
                 after_step()
    guards.py    weight and KV cache refusals, the fail-stop watch
    layers.py    the layer kernels: vLLM tensors -> launches
    runner.py    per-rank launch state: scratch, step epoch, health, peers
    peer.py      peer memory and TP agreement (adapted from ROCm/ATOM, MIT)
mono/            kernel source
    layer.py     K2
    attention/   kda.py (K1), back.py
    stages/      moe.py, gemv.py
    common/      device ops, mailbox sync, MX / int4 helpers, plan / build keys,
                 the target arch (adapted from ROCm/ATOM atom/mono, MIT)
    abi.py       the error-word layout
```
