# Qwen3.5 / Qwen3.6 MoE specialized decode (XPU)

Two decode specializations of the eager XPU Qwen3.5 model (`../model.py`,
`../mtp.py`):

- **Fused MoE** (on by default): the routed and the gated shared experts of a
  decode step of 1-8 rows in one fused kernel (`hooks.py`, `moe.py`,
  `weights.py`). Steps or weights it does not serve run the stock MoE.
- **Mono decode** (opt-in: `--monokernel` or `decode_backend`, below): on two
  Xe2 GPUs at TP2, each decoder layer of a decode or MTP verify step runs as
  one persistent kernel from vllm-xpu-kernels (GatedDeltaNet or full attention,
  then the MoE with the shared expert); both TP all-reduces run inside the
  kernel through mailboxes in the peer GPU's memory. Every other step runs the
  stock layers (with the stock MoE: see below). The rest of this README is
  about the mono decode.

The kernels read vLLM's loaded weights and caches in place; nothing is copied
or re-quantized. The fused MoE would make the shared expert's two fp8
projections row-major [K, N] for its kernels, while the mono kernels read the
row-major [N, K] storage online quantization leaves; so under `--monokernel`
the fused MoE is off, and the steps the mono kernels do not serve (prefill,
mixed, wider and PIECEWISE steps, verify widths without a kernel) run the stock
MoE.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm serve <Qwen3.6-35B-A3B> -tp 2 --dtype float16|bfloat16 \
  --quantization fp8_per_tensor --block-size 64 --monokernel \
  [--speculative-config '{"method": "mtp", "num_speculative_tokens": 1..7}']
```

`--additional-config '{"decode_backend": "latency_specialized"}'` requests it
the same way (the generic switch of the specialized decode RFC,
[#60904](https://github.com/vllm-project/vllm/issues/60904)); either switch is
enough (`runtime.requested`). There are no other options. Without a switch
(`--no-monokernel`, the default) nothing of the mono decode is built or run.

## Supported deployments

Qwen3.5 MoE checkpoints with the 35B-A3B shapes (`runtime.SHAPES`), TP2 on two
Xe2 (Battlemage) GPUs, float16 or bfloat16 activations, online fp8 per-tensor
weights, a KV cache and conv state of the activations' dtype (no float16 /
bfloat16 mix) with a block size that is a multiple of 64, float32 recurrent
state, `max_model_len` up to `_xpu_C.mono_max_context()`, Model Runner V2,
no torch.compile (the eager model), FULL decode graphs (`FULL_DECODE_ONLY` or
`FULL_AND_PIECEWISE`), and no speculative decoding or MTP with 1-7 speculative
tokens. The kernels come from vllm-xpu-kernels built with
`MONO_DECODE_KERNELS_ENABLED` (see [Kernels](#kernels)).

If you request the mono decode for anything else, startup fails with one
`ValueError` that lists every unsupported setting:

- **Config, in the model's `__init__`** (`runtime.unsupported`): other TP sizes,
  pipeline / data / expert / context parallelism, EPLB, DBO, batch-sharded
  sampling, LoRA, KV transfer connectors, sleep mode, CPU weight offload,
  weight updates, routed-expert capture, other dtypes or quantization, KV
  cache or conv state dtypes other than the model's, other recurrent state
  dtypes, other block sizes, other speculative methods or
  token counts, adaptive verification, dynamic speculative decoding, the V1
  model runner, torch.compile, decode graphs other than FULL, other shapes or
  architectures, a `max_model_len` above the kernels' limit, another platform
  or GPU, vllm-xpu-kernels without the kernels.
- **Loaded weights, when the model state builds the executor**
  (`guards.weight_reasons`): fp8 per-tensor linears and experts, norms and
  router in the model dtype, in every decoder layer and the MTP drafter's.
- **Caches, at every bind** (`decode.SpecializedDecode.bind_caches`): the
  kernels check every weight and cache shape, dtype and stride when the layers
  are registered.

Every TP rank agrees on each verdict before anything collective runs: the
kernels wait on the peer GPU, so one rank alone on the mono path would hang the
other.

## Steps

`plan()` runs once per step in the backbone's `forward`. It picks the kernels
for a pure decode step of 1, 2, 4 or 8 rows, or a pure verify step of whole
requests whose rows x tokens have a verify kernel (2x2, 3x3, 4x2, 4x4, 6x3,
8x2, 8x4, 8x8), from host metadata that is the same on every rank. Every other
step runs vLLM's layers: prefill, mixed, wider and PIECEWISE steps, and steps
before a KV cache is bound. Under FULL graphs the plan is baked in at capture;
a replay with fewer requests than its graph pads whole requests, which the
kernels serve like real ones.

The MTP drafter plans its own passes (`plan_draft`): the first pass of a verify
step (the target's rows), then the one-token passes. Its decoder layer and its
two pre-fc norms + fc (`mono_mtp_prologue`) run on the kernels; its lm_head is
the stock one (unquantized, the model dtype).

The executor is built by the model state after load
(`../model_state.Qwen3_5XpuModelState`) and binds the caches in
`initialize_kv_cache`, for the memory-profiling cache and again for the real
one; binding inside graph capture is an error.

## Sampling

The lm_head stays vLLM's unquantized one (the model dtype). On a decode or
verify step of up to 8 rows whose requests are all greedy or sample with a
top_k within
`mono_shard_topk_max_k()` (and need nothing over the whole vocabulary:
logprobs, penalties, logit bias, bad words, allowed tokens, min_tokens, min_p,
structured output, a seed, logits processors), the model state's
`compute_sample_logits` takes each rank's k best logits of its vocabulary shard
and the ranks swap them (`mono_shard_topk`) instead of all-gathering the logits.
vLLM's sampler and rejection sampler (`custom_sampler`) run on these candidates,
which hold every request's top-k, so the result is the stock one; the sampled
positions are mapped back to token ids (`runtime/sampling.py`). Every other
step samples on the full vocabulary.

## Fail-stop

Every in-kernel wait for the peer gives up after about 12 s. The rank records
it in its health words (`_xpu_C.mono_health`; sticky: its later waits exit at
once) and goes on with garbage, so a stalled peer fails every rank instead of
hanging them.

Each step, `compute_sample_logits` queues a non-blocking copy of the rank's
`FAILED` word. Every 16 steps the TP ranks agree on what has landed over the TP
CPU group, and then every rank raises at the same step, so the engine stops.
Detection takes up to about 16 steps, so tokens of the failed steps can reach
the client before the engine stops.

## Kernels

The mono decode needs vllm-xpu-kernels with the mono decode ops: the first
release that contains them (vllm-xpu-kernels PR "[XPU] mono decode", not yet
released), built with `MONO_DECODE_KERNELS_ENABLED=1` (the default build leaves
them out). A build without any of the ops below is refused at startup with the
missing names (`runtime.OPS`).

The op ABI (`torch.ops._xpu_C`; integers that are addresses or handles are
`int`):

```text
mono_decode_supported(int device) -> bool          Xe2 check (-1: current)
mono_max_context() -> int                          longest served context
mono_health(int device) -> Tensor                  int32[8] health words:
                                                   [0] FAILED (sticky),
                                                   [1] timed-out waits
mono_ipc_create(int bytes) -> int                  device region, handle
mono_ipc_export(int handle) -> int[]               IPC handle + dma-buf fd
mono_ipc_open(int handle, int[] peer_info,
              int local_fd=-1) -> int              import the peer region
mono_ipc_ptr(int handle, bool peer) -> int         own / peer address
mono_mbox_bytes() -> int                           bytes of one layer mailbox
mono_register_layer(int layer, bool attn, int mbox,
                    int peer, Tensor[] tensors)    weights + caches of a layer
mono_decode_layer(int layer, Tensor hidden,
                  Tensor residual, Tensor(a!) hid_out,
                  Tensor(b!) res_out, Tensor positions,
                  Tensor m0, Tensor? m1, Tensor? m2) one decode step
mono_decode_layer_spec(int layer, int tokens, ...)
                  -> bool                          one verify step (false:
                                                   no kernel for the shape)
mono_register_mtp(int layer, int mbox, int peer,
                  Tensor[] tensors, int rank)      MTP prologue weights
mono_mtp_prologue(int layer, Tensor x,
                  Tensor(a!) out) -> bool          MTP norms + fc
mono_lmhead_mbox_bytes() -> int                    candidate exchange bytes
mono_shard_topk_max_k() -> int                     largest candidate k
mono_shard_topk(Tensor(a!) vals, Tensor(b!) idx,
                Tensor logits, int k, int rank,
                int mbox, int peer) -> ()          candidates of both shards
```

`mono_register_layer`'s tensor list order and the metadata tensors `m0..m2`
are defined by `runtime/layers.py` and the kernels' `mono_decode_interface.h`;
they change together. Tests only: `mono_fault_inject(int layer, int delta)`
(shifts a layer's mailbox tags: its waits time out).

## Layout

```text
hooks.py         the model-side hooks: the fused MoE's post-load entry,
                 SpecializedDecodeMixin, register_drafter, the model state
moe.py           the fused MoE executor (default)
weights.py       the fused MoE's weight checks
runtime/
    __init__.py  DecodeConfig, specialized_decode_config(), unsupported()
    decode.py    SpecializedDecode: plan() -> Step, Step.layer(i),
                 plan_draft(), bind_caches(), after_step()
    guards.py    weight and bind refusals, the fail-stop watch
    layers.py    the layer and MTP prologue kernels: vLLM tensors -> launches
    peer.py      P2P mailboxes (Level Zero IPC, dma-buf fd exchange), TP
                 agreement
    sampling.py  candidate sampling: the samplers on the vocabulary-shard
                 candidates
../model_state.py  Qwen3_5XpuModelState: builds the executor, binds the caches
```

The model files only call in: `model.py` (the config check in the causal LM's
`__init__`, `plan()` in `Qwen3_5Model.forward`, `step.layer(i)` in the layer
loop, `get_model_state_cls`) and `mtp.py` (`register_drafter`, `plan_draft()`
and the prologue and layer kernels in `forward`). Outside the model:
`vllm/platforms/xpu.py` sets a Level Zero driver key under `--monokernel` (see
the comment there).

Tests: `tests/models/qwen3_5/xpu_specialized_decode/` (vLLM) and
`tests/mono_decode/` in vllm-xpu-kernels (the kernels).
