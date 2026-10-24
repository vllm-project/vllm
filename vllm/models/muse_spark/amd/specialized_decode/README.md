# Muse Spark specialized decode (MI355X)

An opt-in decode path for Muse Spark Open (`MuseSpark1xForCausalLM`,
`MuseSparkForConditionalGeneration`) at TP8 on MI355X (gfx950), following
RFC #60904. On a pure decode step (or a pure DSpark verify step) of at most 8
rows, every decoder layer whose routed experts run AITER's MXFP4 x MXFP4 MoE
becomes one persistent FlyDSL MonoKernel launch (K2: 256 CTAs, one a CU, all
resident), in "layer mode", inside the stock layer loop. The TP all-reduces,
all-gathers and the expert reduce-scatter run in-kernel over peer memory.

K2 runs, in one launch: o_proj and its all-reduce, the attention boundary
(post-attention norm, gated residual blend, pre-FFN norm), the fp32 router and
top-8 routing, the pooler, the pre-expert norm and MXFP4 activation quant, the
routed experts, the column-owner expert reduce, the post-expert norm and the
unpooler with its all-gather. It returns the same deferred FFN boundary as the
stock layer, so kernel and stock layers interleave freely. The kernels read
vLLM's loaded weights in place: nothing is copied or re-quantized.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_MOE_SKIP_PADDING=1 \
vllm serve amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 -tp 8 \
  --attention-backend ROCM_AITER_FA --block-size 16 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --monokernel
```

`--monokernel` and `--additional-config '{"decode_backend": "latency_specialized"}'`
are the same switch. `decode_backend=auto` (the default) without `--monokernel`
runs vLLM's decode; any other `decode_backend` value is an error.

The switch alone serves Phase A: vLLM runs each kernel layer's attention up to
o_proj's input, K2 runs the rest. Two `additional_config` keys move more of the
layer into the kernels:

```bash
--additional-config '{"decode_backend": "latency_specialized",
  "mono_front": "k1", "mono_core": "split"}'
```

| Key | Value | Runs |
| --- | --- | --- |
| `mono_front` | `stock` (default) | vLLM's residual boundary, QKV GEMM and fused QK-norm / RoPE / KV insert |
| | `k1` (B1) | one K1 launch a kernel layer for the same work |
| `mono_core` | `tensor` (default, A) | vLLM's attention and gated norm; K2 from o_proj's input |
| | `gated` (B2) | vLLM's attention; K2 applies the gated norm in its core load |
| | `split` (C) | the split-KV GQA attention core inside K2 at `q = 1` widths; DSpark verify widths keep `gated` (the split does not mask causally per query row) |

An unknown value fails at startup, before anything collective runs; other keys
are ignored. `VLLM_MOE_SKIP_PADDING=1` is required: the kernels read the step's
pad-row mask (`forward_context.is_padding`), which Model Runner V2 refreshes
every step only with that setting.

## Supported deployments

TP8 on MI355X with all 256 CUs, Model Runner V2, the Quark OCP-MXFP4
checkpoint, bf16 model, a bf16 or fp8 (`--kv-cache-dtype fp8`, e4m3) KV cache
(`ROCM_AITER_FA`, unshuffled layout, block size 16), no speculative decoding or DSpark with 7 speculative tokens (static
K), FULL decode CUDA graphs or eager decode. Breakable CUDA graphs are fine.
Multimodal inputs are not refused; `--language-model-only` is the tested
configuration.

Everything else is refused (default-deny): startup fails with one `ValueError`
listing every reason, in three stages, each voted across the TP ranks before
anything collective runs:

- **Config** (the model's `__init__`): any parallelism but TP8 (and TEP8, below), EPLB, DBO, LoRA,
  KV connectors, sleep mode, CPU offload, prompt embeds, speculative methods
  other than static DSpark K = 7, the V1 model runner, PIECEWISE decode graphs,
  a non-bf16 model, an fp8 e5m2 KV cache, an fp8 KV cache with `k1` or `split`
  (the kernels that read or write the cache), another block size, attention backend or
  quantization, another model's shapes or layer pattern, and the platform (not
  ROCm gfx950, FlyDSL missing, fewer than 256 CUs).
- **Loaded weights** (when the model state builds the executor): each kernel
  layer's weight shapes, dtypes and layouts, norm epsilons and gain centres,
  the folded router multiplier, the shuffled MXFP4 experts (TEP8: this rank's
  32 whole experts and their linear expert map); with `k1` or
  `split`, also the qkv / norm weights, RoPE and attention options they read.
- **KV caches** (at every bind): dtype, `[blocks, 2, 16, 128]` layout and the
  resolved decode CUDA-graph mode.

## TEP8 (experts expert-parallel)

`--enable-expert-parallel` on the TP8 deployment (attention TP8, data parallel
1, `--expert-placement-strategy linear`, no EPLB; `mono_core` `tensor` and
`mono_front` `stock` only) builds K2 with `K2Build.moe == "ep"`. Rank r then
holds experts `[32 r, 32 r + 32)` whole (w13 `[32, 8192, 2048]`, w2 `[32, 4096,
2048]`, vLLM's expert-parallel tensors) instead of a 512-column slice of all
256. Only the experts' stages differ (`stages/experts_ep.py`, and `experts.py`
at the expert-parallel geometry `spec.EP_GEOM`):

- the union is the step's picks of this rank's experts (at most 32 slots);
- gate/up is the TP8 stage at the full 4096 intermediate (128 tasks a slot);
  its a2 rows go to `inter_ep` / `isc_ep`, past every other scratch region;
- down: a CTA owns 16 pooled rows as before, but a local pick needs its whole
  a2 row (32 K steps). The a2 rows of the (at most 32 a chunk) local picks are
  staged in LDS in one poll round trip; wave w multiplies four consecutive K
  steps (`4 w` through `4 w` plus 3) of every local slot's w2; the 8 waves'
  fp32 parts are summed in wave order and rounded once to bf16;
- every CTA still pushes all picks' rows to the column owner: a local pick its
  row, any other pick exact zeros, so the owner's fp32 sum over the 8 ranks
  (`stages/reduce.py`, unchanged) is the producing rank's row, and the
  post-expert norm, the route weights and everything after are the TP8 code.

TP8 builds are unchanged (the layout of every TP8 scratch region and every TP8
binary stays). TEP8's expert rows are one fp32 sum rounded once, where TP8 sums
8 bf16-rounded partials, so the two agree to about one bf16 rounding of the
layer's branch (`test_numerics.py::test_k2_tep8_matches_tp8`, relative
difference < 1e-2 against TP8 on the same logical experts). The TEP8 build uses
all 256 VGPRs, so it has the least headroom of the K2 builds.

## Steps, widths and padding

`plan()` runs once a forward. A request decodes `q` rows: 1 without speculative
decoding, 8 with DSpark (K = 7). The built widths are every multiple of `q` up
to 8 rows that the scheduler can reach. A pure decode / uniform verify step of
whole requests at a built width runs the kernels; every other step (prefill,
mixed, wider, PIECEWISE) runs vLLM's layers. Under FULL graphs the plan is baked
in at capture. Pad rows of a padded replay produce exact zeros (selected, never
multiplied by a mask, so NaN / Inf in a pad row cannot leak) and stay out of the
step's expert union. K1 and the split core read MRV2's persistent metadata
buffers; every `plan()` checks their addresses.

## Numerics

- Every cross-rank sum (the all-reduces, the expert reduce, the sums of
  squares) runs in one fixed rank order in fp32, so all ranks' outputs are
  bit-identical. o_proj's all-reduce is rounded once to bf16.
- The router is an fp32 dot; routing (sigmoid, selection bias, lowest index on
  ties, serial renormalization) is the stock kernel's, so ids and weights equal
  stock's on the same logits.
- MXFP4 activations use AITER's RoundUp scale with the `1e-10` amax floor.
- K1 and the gated / split cores reproduce the stock fused QK-norm / RoPE / KV
  insert and `rms_norm_gated` arithmetic; the split core's partial sums are
  merged in base 2 and the attention output is rounded to bf16 as the stock
  core does.
- **Deviation:** the expert rows are reduced over the 8 ranks in fp32 and never
  rounded before the post-expert norm. `VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16`
  changes only the stock path.

The TP8 tests check each stage on its own inputs: router logits within
`rtol=1e-5, atol=1e-6` of `rocm_fp32_router_gemm`, routing `torch.equal` to
`fused_musespark1x_routing`, the pooled rows, finalize and unpooler within 1
bf16 ulp on >= 99.9 % of elements, the layer's stream cos > 0.99999 and branch
cos > 0.9999 against vLLM's ops, every norm at its epsilon and with all-zero
weights, ranks bit-identical, and CUDA-graph replays bit-identical.

## Fail-stop

Every in-kernel wait gives up after a budget (the `WAIT_BUDGET` health word):
120 s from startup and every KV-cache bind until the CUDA-graph capture ends,
then 10 s (120 s throughout under `--enforce-eager`). A rank whose wait gives up
records it in its health words (sticky: its later waits exit at once, addresses
kept in bounds, no K / V written), so a stalled peer fails every rank instead
of hanging them.

Startup failures are voted before the engine reports ready (a self-test launch
of every build at every KV bind, and a vote after capture). While serving,
`compute_logits` queues a non-blocking copy of the rank's `FAILED` word each
step; every 16 steps the TP ranks agree on it and all raise at the same step,
so the engine stops. Tokens of up to ~16 failed steps can reach the client.

Mailbox tags are `(epoch & (2^22 - 1)) * 256 + slot + 1`, with a device step
epoch bumped once a kernel step; nothing is ever zeroed. **Aliasing window:** a
pair a launch does not write (rows past a narrow width, split flags past a
launch's unit count) last written exactly a multiple of 2^22 kernel steps
earlier (about 35 h at 30 ms a step) carries the current tag. This is accepted.

## Residency

K2 and K1 are plain grid launches whose spin waits need all 256 CTAs resident:
one CTA a CU on 256 CUs (refused otherwise), and every build within at most 256
VGPRs, no scratch and 160 KiB of LDS (`test_compile.py` gates every shipped
build). Do not co-schedule other GPU work on the devices.

## Layout

The model files only call in: `amd/model.py` (the config check, `plan()` in
`forward`, `step.layer(i)` in the layer loop, the fail-stop poll in
`compute_logits`, the attention's `front` / `attend` split) and
`amd/model_state.py` (builds the executor after load, binds KV caches, votes
after capture).

```text
hooks.py           SpecializedDecodeMixin: the model-side hooks
runtime/
    __init__.py    the switch, DecodeConfig, the config refusals
    decode.py      SpecializedDecode: plan() -> Step, prepare_kv(), self-test,
                   after_capture(), after_step()
    guards.py      kernel layers, weight and KV cache refusals, the fail-stop vote
    layers.py      Phases (mono_front / mono_core), KernelLayer: vLLM tensors
                   -> K1 / K2 launches
    runner.py      per-rank launch state: scratch, step epoch, health, peers
    peer.py        peer memory and TP agreement (adapted from ROCm/ATOM, MIT)
mono/              kernel source
    spec.py        the shared contract: shapes, placements, build keys,
                   argument lists, scratch and peer layouts
    abi.py         the health-word layout
    layer.py       K2 (cores: tensor, gated, split)
    k1.py          K1
    front/         K1's stages: boundary, qkv GEMV, epilogue
    attention/     o_proj + all-reduce push, the gated norm, the split-KV core
    stages/        seam, router, pool, experts, reduce, unpool, gemv
    common/        device ops, mailbox sync, MX helpers, build keys, the
                   route table and stage context (adapted from ROCm/ATOM
                   atom/mono, MIT, via the Kimi-K3 specialized decode)
```

Tests: `tests/models/muse_spark/amd_specialized_decode/` (`test_runtime.py`
on CPU; `test_compile.py` with FlyDSL for gfx950, GPU-less; the TP8 and TEP8
`test_numerics*.py` on MI355X).
