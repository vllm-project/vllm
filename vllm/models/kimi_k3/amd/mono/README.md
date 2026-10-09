# Kimi-K3 mono decode (MI355X)

Opt-in with `VLLM_ROCM_MONO_DECODE=1`. A spec-verify step of at most 8 rows
runs each decoder layer as two persistent FlyDSL launches (256 CTAs, all
resident), the TP all-reduces in-kernel over peer memory:

- **K1** (`attention/kda.py`, KDA layers): AttnRes, in_proj, f_b, spec conv,
  KDA recurrence, gated RMSNorm.
- **K2** (`layer.py`, MoE layers): `attention/back.py` (o_proj + all-reduce +
  MLP AttnRes) then `stages/moe.py` (router, top-16, MXFP4 experts, shared
  experts, latent all-reduce + RMSNorm, up_proj, final all-reduce).

```text
amd/mono_decode.py   eligibility per layer / per step, vLLM tensors -> kernel inputs
amd/mono/
    runner.py        scratch, step epoch, peer memory
    layer.py         K2
    attention/       kda.py (K1), back.py
    stages/          moe.py, gemv.py
    common/          device ops, mailbox sync, MX helpers, plan / build keys,
                     peer memory, debug (adapted from ROCm/ATOM atom/mono, MIT)
```

Constraints: TP8, gfx950, at most 8 rows a step; K1 needs breakable CUDA
graphs. Off with a KV connector: the kernels spin-wait on every CTA being
resident, and a connector's copy kernels on another stream can hold the CUs.
Any other step takes the existing path.
