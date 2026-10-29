# Mistral Large 4 specialized decode (MI350X / MI355X)

An opt-in decode path for Mistral Large 4 at TP4 on gfx950, following RFC
#60904. A pure decode step of 1, 2 or 4 requests (one token each) runs
every MoE decoder layer (1-71) as one persistent FlyDSL MonoKernel launch (256
CTAs, all resident), in "step mode": the specialized decode runs the whole
forward. Each launch covers the input
RMSNorm, the q / latent-KV projections with SSMax and LeRoPE, the DSA indexer
(projections, key cache write, scoring and exact top-2048), sparse MLA, o_proj,
the router (softmax top-4), the shared and routed MXFP4 experts and both TP
all-reduces (in-kernel, over peer memory). The embedding, the dense layer 0 and
the final norm run vLLM's modules.

The kernel is ROCm/ATOM's GLM-5.2 mono kernel adapted to the ML4 layer: no
q-LoRA, SSMax on the query and the index query, LayerNorm'd index inputs,
softmax routing, NoPE layers, KV-sharing layers (39, 43, ..., 71 attend to
layer 35's latent cache) and index-sharing layers (4k+1, 4k+2 for k >= 4 reuse
the top-k of layer 4k).

## Enabling

```bash
VLLM_FP8_MOE_REQUANT=mxfp4 VLLM_FP8_MOE_REQUANT_ACT=a4w4 \
vllm serve <Mistral-Large-4> -tp 4 --dtype bfloat16 \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

`--monokernel` enables it too on builds whose kernel config has that switch.
`decode_backend=auto` (the default) runs vLLM's decode. There are no other
options.

## Supported deployments

TP4 on gfx950 with all 256 CUs; Model Runner V2; a latent KV cache in FP8
(E4M3 with KV scale 1.0, the ROCm default) or bf16, with a block size that is a
multiple of 16 (the DSA indexer's FP8 key cache shares its block ids); routed
experts in AITER's a4w4 MXFP4 storage (`AITER_MXFP4_MXFP4`,
e.g. the block-FP8 checkpoint re-quantized at load as above). The kernels read
the routed experts and both caches in place. The attention, indexer and router
weights are repacked into the kernels' tiled formats (4.17 GiB per rank) and
the shared expert is re-quantized to MXFP4 with the same quantizer; this
happens after load, before vLLM's memory profiling, so the KV cache is sized
with it.

If the specialized decode is requested for anything else, startup fails with
one `ValueError` listing every reason (default-deny):

- **Config, in the model's `__init__`** (`runtime/spec.py`): pipeline, data,
  expert, decode- or prefill-context parallelism, EPLB, DBO / ubatching, LoRA,
  KV connectors or offloading, routed-expert capture, speculative decoding
  other than DSpark (or dynamic, or more than 11 speculative tokens), sleep mode, weight transfer, CPU weight offload, stock torch.compile, the V1
  model runner, a KV cache dtype other than FP8 (E4M3) or bf16, a block size
  that is not a multiple of 16, block-FP8 experts not re-quantized to a4w4.
- **Loaded model, when the model state builds the decode** (`runtime/guards.py`):
  the layer mix (one dense layer, then MoE layers whose kinds the kernels
  build), each kernel layer's weight dtypes and shapes, the MoE backend,
  ROCm / FlyDSL / gfx950 / 256 CUs.
- **KV scales, after load** (`runtime/runner.py`): an FP8 latent cache whose
  KV scale is not 1.0 (the kernels read and write it unscaled).
- **Caches, at the first planned step**: each layer's latent and indexer cache
  dtype, shape and contiguity, and the shared block ids.

Every TP rank agrees on each verdict before anything collective runs.

## Steps

`plan()` runs once per forward. It picks the kernels for a pure decode step
whose row count is one of the built widths, 1, 2 or 4 (`DecodeConfig.widths`:
vLLM's FULL-graph capture sizes, so graph replays pad decode batches to them;
padded rows attend to no keys and write nothing). Every other step runs vLLM's
layers: prefill, mixed, wider (5+ requests), eager steps of other sizes and
microbatched steps. The decisions read host metadata every rank holds alike.
One kernel is built per width and layer kind (12 launchers) at the first
planned step. An 8-row launch exists and is correct, but measured slower than
vLLM's layers at 8 requests, so it is not served.

With DSpark (`--speculative-config '{"method": "dspark", ...}'`), a target
verify step is `q = num_speculative_tokens + 1` consecutive rows per request
(the kernels' `index_request_width`; each row's causal extent comes from its
position). The kernels take a pure decode step of one request's block, `q`
rows (`DecodeConfig.q`, `widths = (q,)`); two requests (8 rows at `q = 4`)
measured slower than vLLM's layers and run them, as do larger steps. The
residual stream at the target's `aux_hidden_state_layers` is cloned between
launches for the DSpark draft.

## Fail-stop

Every in-kernel wait gives up after 10 s, and giving up sets a sticky fault word
on every rank (in each rank's symmetric peer buffer), whose own waits then end
at once. After every forward, mono or stock, eager or replayed, before logits, a
device-side assert (`torch._assert_async`) checks the word. A set word traps the
GPU queue: none of the step's later work runs, the worker aborts and the engine
stops. No token of a failed step reaches a client. The check costs about 1 us
per launch (the word and the clock are read every 32 polls of a wait).

## Layout

Everything lives in this folder. The model files only call in: `model.py` (the
config check in `MistralLarge4Model.__init__`, `plan` / `run` / `after_step` in
`forward`, `get_model_state_cls`) and `../model_state.py` (builds the decode
after load). `pixtral.py` forwards `get_model_state_cls` from its text model.

```text
runtime/
    __init__.py  DecodeConfig, specialized_decode_config()
    spec.py      the switch and the config refusals (side-effect free)
    guards.py    loaded-model, device and cache refusals
    decode.py    SpecializedDecode: plan() -> Step, run(), after_step(), close()
    runner.py    per-rank launch state: weights, launchers, scratch, peers, fault word
    weights.py   vLLM modules -> kernel arguments; layer kinds
    variants.py  the kernel package loaded once per layer kind (RoPE / NoPE / NoPE+KV-share)
mono/            kernel source (adapted from ROCm/ATOM atom/models/glm52/mono, Apache-2.0)
    kernel.py    the layer kernel; layout.py, geometry*.py, op.py (host wrapper)
    common/      device ops, mailbox sync, peer memory, weight packing
                 (adapted from ROCm/ATOM atom/mono, MIT)
```

Tests: `tests/models/mistral_large4/amd_specialized_decode/` (CPU: switch,
refusals, layer kinds, step routing).
