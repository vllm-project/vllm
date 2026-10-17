# Kimi-K3 mono decode (MI355X)

A MonoKernel as RFC #60904 proposes, opt-in with `--monokernel`. A pure
spec-verify step of at most 8 rows runs each MoE decoder layer as persistent
FlyDSL launches (256 CTAs, all resident), the TP all-reduces in-kernel over
peer memory:

- **Whole layers** (KDA): K1 (AttnRes, in_proj, f_b, spec conv, KDA recurrence,
  gated RMSNorm), then K2.
- **FFN layers** (MLA): vLLM's AttnRes and MLA up to o_proj's input, then K2.
- **K2**: o_proj + all-reduce + MLP AttnRes, then the latent MoE (router,
  top-16, MXFP4 experts, shared experts, latent all-reduce + RMSNorm, up_proj,
  final all-reduce).

```text
mono/
    spec.py      MONO_SPEC: layers, widths, supports, refusals, pack, bind,
                 eligible, build
    kernel.py    the layer kernels: vLLM tensors -> launches
    runner.py    per-rank launch state: scratch, step epoch, health, peers
    runtime.py   model-agnostic: configure, attach, forward_layer, health
                 (private until vllm/models/common/mono/ exists)
ops/mono/        kernel source
    layer.py     K2
    attention/   kda.py (K1), back.py
    stages/      moe.py, gemv.py
    common/      device ops, mailbox sync, MX helpers, plan / build keys,
                 peer memory (adapted from ROCm/ATOM atom/mono, MIT)
```

With the switch on, an unsupported config raises at startup with every reason.
Every other step (prefill, mixed, other widths, PIECEWISE graphs) runs vLLM's
layers.

Every in-kernel wait gives up after 10 s: the rank records the wait, raises its
flag on the output rank and stops waiting. `step_health()` returns the output
rank's flags; Model Runner V2 fails the step before its tokens are returned.
`tests/models/kimi_k3/amd_mono/test_engine_fault.py` stalls a rank in a served
model and checks the engine stops.
