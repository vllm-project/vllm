# RFC / exploration: compile the decode step end-to-end, including KV-cache management

Status: **draft / RFC** (exploration + measurement, not a merge-ready feature).
Builds on the stock `torch.compile` migration (#46423). Authored with AI assistance.

## Question

#46423 moves GPT-OSS onto stock `torch.compile`. Natural next question: can we
compile the *whole* decode step end-to-end -- not just the dense forward, but the
KV-cache management too (block tables, slot mapping, attention-metadata build)?

## What is compiled/captured today (the map)

Two orthogonal mechanisms, kept distinct:

- **Compile partitioning** -- how Inductor splits the FX graph (at attention ops
  via `splitting_ops` / graph partition).
- **Cudagraph capture** -- an external wrapper records a `torch.cuda.CUDAGraph`
  around a runnable; FULL captures the whole forward, PIECEWISE captures per
  non-attention partition.

Key facts (all on `bobren/vanilla-torch-compile`):

- Attention and the KV write are **opaque custom ops** in *every* mode --
  `unified_attention_with_output` / `unified_kv_cache_update` are registered via
  `direct_register_custom_op` with fake impls (`attention.py:747-805`). Inductor
  never traces into them; it uses only their meta impls. FULL vs PIECEWISE differs
  only in whether a cudagraph is recorded *around* them, never whether Inductor
  compiles *through* them.
- For **FA3 models like GPT-OSS** (`AttentionCGSupport.ALWAYS`,
  `flash_attn.py:313-317`), **decode already captures the entire forward --
  attention and the KV-cache write included -- in a single FULL cudagraph** via
  `StockTorchCompileCUDAGraphWrapper(runtime_mode=FULL)` over the `fullgraph=True`
  compiled model (`gpu_model_runner.py:5325-5330`, `stock_cudagraph.py:202-221`).
  So "attention + KV in one graph for decode" is the shipped state, not a frontier.
- What is **NOT** compiled/captured is the per-step **host-side KV-cache
  management + input prep**, run eagerly in Python before every forward:
  `_prepare_inputs` (block-table H2D commit, the slot-mapping Triton kernel,
  positions / seq-len math; `gpu_model_runner.py:1890-2215`) and
  `_build_attention_metadata` (block-table gather, slot-mapping fetch, per-layer
  metadata; `2217-2518`). The opaque attention op reads block tables / slot mapping
  / seqlens out-of-band from a module-global forward context.

## The prize (measurement)

`benchmarks/compile/decode_step_breakdown.py` profiles a decode workload and
attributes per-step CPU time. GPT-OSS-120B, H100 TP1, stock path (FULL_AND_PIECEWISE):

| per-decode-step CPU (us)        | batch=1 | batch=32 |
|---|---:|---:|
| `_prepare_inputs`               | 3255 | 3636 |
| `_build_attention_metadata`     |  748 |  719 |
| `_preprocess`                   |   93 |   88 |
| **host KV-mgmt + input prep**   | **4096** | **4442** |
| `_model_forward` (captured)     |  819 |  944 |
| `sample_tokens`                 | 1108 | 1251 |
| **host-prep / forward ratio**   | **5.0x** | **4.7x** |

The host-side KV-cache management + input prep costs **~5x the CPU of the
already-captured forward, at both batch 1 and batch 32** -- and it barely grows
with batch, because the forward is a cheap cudagraph replay while the prep is
fixed Python/numpy overhead. **This is the frontier**: the forward is solved;
the un-compiled remainder is the KV-cache management.

Caveat: `with_stack` profiling inflates Python-heavy regions, so the absolute us
and the exact ratio are overstated; read the direction, not the digits. In a live
server this host prep is partly hidden by async scheduling, but it becomes the
ceiling exactly as GPU work shrinks (small batch, high TP, faster kernels).

## Directions (and why the obvious one is already done)

- **A. Extend FULL cudagraph capture to prefill/mixed.** GPT-OSS is FA3 `ALWAYS`,
  so mixed FULL is only blocked by cascade attention (`backend.py:549-551`) and
  capture-size padding. Low risk, but the launch-overhead win is a decode effect;
  prefill is compute-bound, so the payoff is small and it costs capture memory.
- **B. Compile / tensorize the host KV-cache management + input prep, and capture
  its kernels into the step (recommended -- this is the prize).** Two sub-steps:
  1. *Capture the GPU-side KV-management kernels* (slot-mapping, block-table copy)
     into the decode graph by running them from static buffers inside the captured
     runnable. They already write persistent buffers the attention op reads, so
     the aliasing makes this tractable -- but it only reclaims the ~0.2-0.5 ms of
     kernel-launch, not the Python.
  2. *Tensorize the host prep* so the ~3-4 ms of `_prepare_inputs` /
     `_build_attention_metadata` Python/numpy becomes a small, static, capturable
     tensor program driven from staged scheduler outputs. This is the real win and
     the real work.
- **C. Generalize the breakable-cudagraph whole-forward path**
  (`breakable_cudagraph.py`, behind `VLLM_USE_BREAKABLE_CUDAGRAPH`) to cover
  prefill with eager breaks at attention -- an alternative to A that sidesteps FX
  splitting.

## Top risks for the prize (B)

1. **Host prep is data-dependent Python.** `_prepare_inputs` /
   `_build_attention_metadata` are per-request numpy loops (`2118-2131`,
   `2217-2518`); not Dynamo-traceable as-is. Needs a tensorized, branch-free rewrite
   driven from staged scheduler outputs.
2. **Static-buffer invariant.** slot_mapping / block_table / seq_lens must occupy
   fixed addresses across replays (the wrappers assert address stability in DEBUG,
   `cuda_graph.py:346-355`); folding their *computation* into the graph means their
   *inputs* (scheduler output) must be staged into static buffers first.
3. **Cross-cutting features assume eager prep** -- LoRA, spec-decode metadata,
   KV-connector transfers, DP/ubatch coordination, mamba state indices each mutate
   host/device state during prep and need graph-safe handling.

## What is in this draft

- `benchmarks/compile/decode_step_breakdown.py` -- the reproducible measurement
  above (uses the built-in torch profiler; no runtime changes).
- This RFC.

Deliberately **not** included: a half-working capture prototype. The measurement
redirects the naive plan ("capture attention+KV into a graph" -- already shipped
for decode) toward the real prize (tensorize + capture the host KV-management),
which is a substantive change worth designing before coding. Proposed next step:
prototype B.1 (slot-mapping/block-table kernels inside the decode capture) behind
an opt-in flag to validate the mechanism, then scope B.2.
