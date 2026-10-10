# Kimi-K3 mono decode (MI355X)

Opt-in with `VLLM_ROCM_MONO_DECODE=1`. A spec-verify step of at most 16 rows
runs each decoder layer as two persistent FlyDSL launches (256 CTAs, all
resident), the TP all-reduces in-kernel over peer memory. For 17..64 rows,
KDA layers use K1 followed by the existing projection and AITER MoE path:

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

Rows are processed in tiles of at most 16 inside each persistent launch. K1
tiles keep complete requests together; each request has at most 8 verification
tokens. LDS is sized for one tile. K1 scratch regions are separate for
each tile, so workers can advance without overwriting another tile's mailbox.
For more than 8 rows, K1 stages 4096 then 3072 input columns in LDS and keeps
both projection groups' accumulators in registers. Each weight slice serves
all token tiles before the next slice is loaded. K2/MoE stages wide GEMVs in
4096-column slices and uses a bounded expert-to-token table for each tile;
K2/MoE runs only one tile.

K1 groups 32 state rows per KDA task for 24 rows with verification length 6,
and 64 state rows for 48/4 and 50/5. Serving also selects 32 state rows for
32/8 and 64 for 64/8. Each thread holds multiple state rows in registers and
shares the gates and q/k normalization between them. Other shapes retain
16 state rows per task.

K1 caches the callable returned by `flydsl.compiler.compile` per build.
Metadata pointers, strides, layer tags, and streams are runtime arguments;
the compile call itself executes the first launch. This avoids repeating
FlyDSL's signature and JIT-cache analysis at each eager graph break.

K2 and standalone mono MoE support up to 16 rows in one tile. Larger batches
use the production projection and AITER MoE path because persistent MoE
regresses at larger row counts. K1 remains eligible up to 64 rows. M64 has not
shown a stable end-to-end gain at conc8. Performance results apply to
TP8/DCP1 with DSpark k=7; other parallelism and verification lengths require separate measurement.

Constraints: TP8, gfx950, at most 64 rows for K1 and 16 for K2/MoE; K1 needs
breakable CUDA graphs. Off with a KV connector: the kernels spin-wait on
every CTA being resident, and a connector's copy kernels on another stream can hold the CUs.
Any other step takes the existing path.
