# RFC / exploration: compile the decode step end-to-end, including KV-cache management

Status: **draft / RFC** (exploration + measurement, not a merge-ready feature).
Builds on the stock `torch.compile` migration (#46423). Authored with AI assistance.

## Question

PR #46423 moves GPT-OSS onto stock `torch.compile`. The natural next question: can
we compile the *whole* decode step end-to-end -- not just the dense forward, but the
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
  compiled model (`gpu_model_runner.load_model`, `StockTorchCompileCUDAGraphWrapper.__call__`).
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

| per-decode-step CPU (us)      |  batch=1 | batch=32 |
| ----------------------------- | -------: | -------: |
| `_prepare_inputs`             |     3255 |     3636 |
| `_build_attention_metadata`   |      748 |      719 |
| `_preprocess`                 |       93 |       88 |
| **host KV-mgmt + input prep** | **4096** | **4442** |
| `_model_forward` (captured)   |      819 |      944 |
| `sample_tokens`               |     1108 |     1251 |
| **host-prep / forward ratio** | **5.0x** | **4.7x** |

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

## B.1 prototype (this PR): slot-mapping in the decode cudagraph

Behind opt-in `VLLM_STOCK_CAPTURE_KV_PREP`, the per-step slot-mapping Triton kernel
is folded into the stock FULL decode cudagraph via a pre-forward hook on
`StockTorchCompileCUDAGraphWrapper`, instead of being launched eagerly every step.
The kernel writes the persistent `slot_mapping` buffer the attention op reads
(aliasing), ordered before the forward inside the capture; each replay recomputes
`slot_mapping` from the per-step-refreshed persistent buffers. It is gated (in
`initialize_kv_cache`, once the real per-group block tables + config are known) to
the verified-safe scope: all-attention KV-cache groups (GPT-OSS's full +
sliding-window groups qualify; Mamba/conv rejected), no spec-decode / mrope / xdrope
/ context-parallel / routed-experts snapshot. Non-FULL and non-uniform steps issue
slot-mapping eagerly as before; the kernel is warmed at init so its first Triton JIT
is not recorded inside a capture.

**Validation.**

*Correctness -- byte-identical, with the capture proven to fire.* Two suites, both
comparing greedy token ids flag-off vs flag-on and both asserting (via a compilation
counter) that the in-graph capture hook actually fired -- so a silent gate fallback
cannot make off==on vacuously.

- **Committed CI test** (`tests/compile/test_stock_capture_kv_prep.py`, opt-125m, a
  deterministic all-attention proxy): flag-off == flag-on token ids across five
  KV-cache regimes -- padded decode (R < captured graph size, batch 3->4 and 7->8),
  long multi-block sequences, shared-prefix block reuse, a batch-size sweep, and block
  churn across many requests -- with chunked prefill enabled, in **both**
  `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`. Off/on run in one process (shared
  JIT/autotune) with flashinfer autotune disabled; the hook fires (counter > 0) only
  when the flag is on. Passes (`2 passed`).
- **Target-model parity** (GPT-OSS-120B, H100 TP1, FA3, two attention groups): across
  the same five KV regimes, flag-on is **byte-identical** to flag-off, and a
  flag-off-vs-flag-off control is byte-identical too -- so the divergence the prototype
  adds is zero, not merely below the model's own run-to-run noise. The hook fired into
  **83** decode captures under the flag (0 with it off). (Each run is a fresh process
  with flashinfer autotune off; the "not reproducible across invocations" effect was a
  measurement artifact of autotune, not the model.)

*Mechanism.* The log confirms `slot-mapping folded into the FULL decode cudagraph`
and the hook records into all 83 GPT-OSS decode captures.

*CPU breakdown (with_stack profiler; read the direction, not the digits):* the eager
`compute_slot_mapping` launch is eliminated from the decode path --

| eager per-step region  | batch=1 off->on              | batch=32 off->on |
| ---------------------- | ---------------------------- | ---------------- |
| `compute_slot_mapping` | 132 -> **0** us (n=192 -> 0) | 142 -> **0** us  |
| `_prepare_inputs`      | 2986 -> 2720 us              | 3556 -> 3246 us  |

*Competitiveness (interleaved 3-way, `vllm bench serve`, 4 launches per backend
round-robin so autotune/thermal drift is balanced; random 1024-in/128-out; 3 reps per
launch).* `stock_on` (this PR) vs the two shipping GPT-OSS decode paths -- stock
eval-frame without the flag (`stock_off`) and legacy `VllmBackend` mode=3 (`vllm`):

| metric (GPT-OSS-120B, H100 TP1, n=4)          | stock_off      | stock_on (PR)  | vllm (mode=3)  |
| --------------------------------------------- | -------------- | -------------- | -------------- |
| saturated tok/s                               | 2432 +/- 79    | 2501 +/- 38    | 2401 +/- 181   |
| batch-1 ITL (ms)                              | 5.09           | 5.10           | 5.08           |
| mid-batch(16) ITL (ms)                        | 13.03 +/- 0.15 | 13.10 +/- 0.14 | 13.35 +/- 0.65 |

`stock_on` does **not regress `stock_off` on any metric** -- throughput 1.028x (Welch
t=+1.6, n.s.), batch-1 ITL 1.000x (t=+0.5, n.s.), mid-batch ITL 1.005x (t=+0.7, n.s.)
-- and is competitive with `VllmBackend`: throughput 1.042x and mid-batch ITL 0.981x
(both n.s.), batch-1 ITL 1.004x (t=+8.1, but that is 0.02 ms / 0.4% with near-zero
variance and it equals `stock_off`, so it is a stock-vs-`VllmBackend` baseline gap, not
introduced by this PR). Expected: the ~130 us/step of launch CPU removed is largely
overlapped by async scheduling, and the ~3 ms of non-capturable host Python (the real
prize) is untouched.

So B.1 validates the *mechanism* -- KV-cache management can be captured into the
decode graph byte-identically, on both a deterministic proxy and the target model,
without regressing e2e performance vs either shipping path -- and confirms the
step-time win lives in B.2 (tensorizing the host prep), not in capturing the kernel.

## What is in this draft

- `benchmarks/compile/decode_step_breakdown.py` -- the reproducible measurement
  above (uses the built-in torch profiler; no runtime changes).
- The B.1 prototype (`VLLM_STOCK_CAPTURE_KV_PREP`, opt-in, off by default).
- `tests/compile/test_stock_capture_kv_prep.py` -- the committed byte-identical CI
  test (both FULL cudagraph modes, five KV regimes, capture-hook-fired assertion).
- This RFC.

Proposed next step: **B.2** -- tensorize `_prepare_inputs` /
`_build_attention_metadata` into a static, capturable tensor program driven from
staged scheduler outputs, so the ~3 ms of host Python (which dominates the decode
step, ~5x the captured forward) can itself be folded into the graph. B.1 establishes
the capture + persistent-buffer aliasing mechanism that B.2 builds on.
