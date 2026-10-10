# Kimi-K3 mono decode (MI355X, MI300X / MI325X)

Opt-in with `VLLM_ROCM_MONO_DECODE=1`. A spec-verify step of at most 8 rows
runs each decoder layer as two persistent FlyDSL launches (256 CTAs, all
resident), the TP all-reduces in-kernel over peer memory:

- **K1** (`attention/kda.py`, KDA layers): AttnRes, in_proj, f_b, spec conv,
  KDA recurrence, gated RMSNorm.
- **K2** (`layer.py`, MoE layers): `attention/back.py` (o_proj + all-reduce +
  MLP AttnRes) then `stages/moe.py` (router, top-16, routed experts, shared
  experts, latent all-reduce + RMSNorm, up_proj, final all-reduce).

```text
amd/mono_decode.py   eligibility per layer / per step, vLLM tensors -> kernel inputs
amd/mono/
    runner.py        scratch, step epoch, peer memory
    layer.py         K2
    attention/       kda.py (K1), back.py
    stages/          moe.py, gemv.py
    common/          device ops, mailbox sync, MX / int4 helpers, plan / build
                     keys, peer memory, debug (adapted from ROCm/ATOM atom/mono, MIT)
```

The build targets the device's arch (`FLYDSL_GPU_ARCH` overrides it):

| | gfx950 (MI355X) | gfx942 (MI300X / MI325X) |
|---|---|---|
| routed experts | MXFP4 as loaded, MXFP4 latent / INTER | vLLM's packed-int4 requant (`--quantization-config.moe.weight int4_per_group_32`), bf16 latent / INTER |
| LDS | 160 KB: whole x rows | 64 KB: x a K window at a time, the ug latent and the down INTER in K phases |
| MFMA | 16x16x32 bf16, cross-lane by permlane | two 16x16x16 bf16, cross-lane by ds_swizzle / ds_bpermute |

Constraints: TP8, gfx950 or gfx942 with at least 256 CUs (no CPX partition),
at most 8 rows a step; K1 needs breakable CUDA graphs. On gfx942 without the
int4 expert requant only K1 runs. Off with a KV connector: the kernels
spin-wait on every CTA being resident, and a connector's copy kernels on
another stream can hold the CUs. Any other step takes the existing path.
