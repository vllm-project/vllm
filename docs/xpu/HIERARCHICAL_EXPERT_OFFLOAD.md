# Hierarchical MoE expert staging on XPU

See the full user guide:
[`docs/features/hierarchical_expert_offload.md`](../features/hierarchical_expert_offload.md).

## XPU-specific notes

- Prefer **explicit DMA** (`tensor.copy_(..., non_blocking=True)` on an XPU
  copy stream) into device expert slots. Do **not** rely on UVA host-pointer
  kernel loads for expert GEMMs.
- Disk→RAM uses `O_DIRECT` into already-pinned frames (no `cudaHostRegister`
  analogue required for v1).
- Multi-stream compute∥copy overlap can be weaker than CUDA; deepen
  `--tier-pilot` / raise `--tier-num-slots` if stalls dominate.
- Graphs remain **experimental** (`--tier-allow-cuda-graphs`). Default path
  forces `enforce_eager`.
- Mutual exclusion with EPLB in v1.

## Bakeoff

Default hardware model: **Mixtral-8x22B Instruct AWQ (Q4)**
(`MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ`, local path
`/tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ`). Mixtral has 8 experts;
use `--tier-num-slots 4` to force RAM↔device staging.

```bash
python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 \
  --tier-ram-gb 32 \
  --colibri-tok-s <published_colibri_number> \
  --output /tmp/tier_bakeoff.json
```
