# Kimi-K3 mono decode (MI355X)

A MonoKernel as RFC #60904 proposes, opt-in with `--monokernel`. A pure
spec-verify step of at most 16 rows runs each MoE decoder layer as persistent
FlyDSL launches (256 CTAs, all resident), the TP all-reduces in-kernel over
peer memory. The kernels tile up to 64 rows, but at 32 and 64 rows they were
slower than vLLM's path, so the spec caps a step at 16 (`MAX_ROWS`):

- **Whole layers** (KDA): K1 (AttnRes, in_proj, f_b, spec conv, KDA recurrence,
  gated RMSNorm), then K2.
- **FFN layers** (MLA, TP8 + DCP8 or DCP1): M1a (AttnRes, qkv_a and g_proj,
  q / kv RMSNorm, the fp8 KV write of this rank's tokens, q_b, MXFP4 W_UK, the fp8
  query gathered to every rank's peer memory), AITER's MLA decode on this rank's
  KV shard, M1b (the a2a LSE combine of every rank's partials, MXFP4 W_UV, the
  sigmoid gate), then K2. At DCP1 (every rank holds the whole KV) M1a writes
  its 12 heads' query directly and M1b reads AITER's output as is: no gather,
  no combine, no M1 peer memory.
- **K2**: o_proj + all-reduce + MLP AttnRes, then the latent MoE (router,
  top-16, MXFP4 experts, shared experts, latent all-reduce + RMSNorm, up_proj,
  final all-reduce).

```text
mono/
    spec.py      MONO_SPEC: layers, widths, supports, refusals, pack, bind,
                 eligible, build
    kernel.py    the layer kernels: vLLM tensors -> launches
    runner.py    per-rank launch state: scratch, step epoch, health, peers
    mla.py       the FFN layers' MLA as vLLM ops: the context, AITER's
                 decode (the M1a / M1b test reference)
ops/mono/        kernel source
    layer.py     K2
    attention/   kda.py (K1), mla.py (M1a), mla_back.py (M1b), back.py
    stages/      moe.py, gemv.py
    common/      device ops, mailbox sync, MX helpers, plan / build keys,
                 peer memory (adapted from ROCm/ATOM atom/mono, MIT)
```

Every launch's mailbox tag is the step epoch and a slot: K1 and M1a / M1b
use the layer index, K2 128 + the layer index. The peer buffer holds K2's
regions, then M1a's gathered query, then M1b's partials (both DCP8 only); a
region is reused by the next layer only after K2's all-reduce, which every
rank reaches past its M1b.

With the switch on, an unsupported config raises at startup with every reason.
Every other step (prefill, mixed, other widths, PIECEWISE graphs) runs vLLM's
layers.

Every in-kernel wait gives up after 10 s: the rank records the wait, raises its
flag on the output rank and stops waiting. `step_health()` returns the output
rank's flags; Model Runner V2 fails the step before its tokens are returned.
`tests/models/kimi_k3/amd_mono/test_engine_fault.py` stalls a rank in a served
model and checks the engine stops.
