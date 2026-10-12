# Kimi-K3 mono decode (MI355X)

Opt-in with `VLLM_ROCM_MONO_DECODE=1`, laid out as RFC #60904 proposes. A pure
spec-verify step of at most 8 rows runs each MoE decoder layer as persistent
FlyDSL launches (256 CTAs, all resident), the TP all-reduces in-kernel over
peer memory:

- **Whole layers** (KDA): K1 (`attention/kda.py`: AttnRes, in_proj, f_b, spec
  conv, KDA recurrence, gated RMSNorm), then K2.
- **FFN layers** (MLA): vLLM's AttnRes and MLA up to o_proj's input, then K2.
- **K2** (`layer.py`): `attention/back.py` (o_proj + all-reduce + MLP AttnRes),
  then `stages/moe.py` (router, top-16, MXFP4 experts, shared experts, latent
  all-reduce + RMSNorm, up_proj, final all-reduce).

```text
spec.py      kernel layers, widths, refusals (config / model / caches)
routing.py   attach, begin_step, forward_layer, health
kernel.py    the layer kernels: vLLM tensors -> launches
runner.py    scratch, step epoch, health words, peer memory
layer.py     K2
attention/   kda.py (K1), back.py
stages/      moe.py, gemv.py
common/      device ops, mailbox sync, MX helpers, plan / build keys,
             peer memory, debug (adapted from ROCm/ATOM atom/mono, MIT)
```

With the switch on, an unsupported config raises at startup with every reason
(`spec.refuse_config`). Every other step (prefill, mixed, other widths,
PIECEWISE graphs) runs vLLM's layers.

Every in-kernel wait gives up after 10 s: the rank records the wait, raises its
flag on the output rank and stops waiting. `step_health()` returns the output
rank's flags; Model Runner V2 fails the step before its tokens are returned.
`VLLM_MONO_FAULT_INJECT=rank:epoch:us` (tests only) stalls a rank.
