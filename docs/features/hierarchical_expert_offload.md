# Hierarchical (Colibri-style) MoE Expert Offloading

vLLM can stage Mixture-of-Experts **routed expert weights** across a
three-tier hierarchy inspired by [Colibrì](https://github.com/JustVugg/colibri):

```text
NVMe ExpertStore  →  pinned system RAM (LFRU + learned pins)  →  device slots (XPU/CUDA)
```

**Placement only affects speed** — never precision or router semantics.

Dense components (attention, embeddings, norms, gates, shared experts, LM head)
stay resident on device. Only routed experts (`w13_weight` / `w2_weight` and
scales) are streamed.

## When to use it

- MoE models that do not fit entirely in device memory
- Intel **XPU** (Xe2/Xe3) primary target; CUDA works for the RAM↔device path
- Models larger than RAM when `--tier-disk-path` points at an ExpertStore

## Quick start

```bash
# Hardware bakeoff default: Mixtral-8x22B Instruct AWQ (Q4)
vllm serve MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ \
  --offload-backend hierarchical \
  --tier-num-slots 4 \
  --tier-ram-gb 32 \
  --tier-policy quality \
  --enforce-eager
```

Force a disk tier (builds ExpertStore on first run if missing):

```bash
vllm serve <moe-model> \
  --offload-backend hierarchical \
  --tier-device-expert-gb 4 \
  --tier-ram-gb 16 \
  --tier-disk-path /nvme/expert_store \
  --tier-pilot \
  --tier-policy balanced \
  --tier-repin-tokens 64
```

## CLI reference

| Flag | Meaning |
|------|---------|
| `--offload-backend hierarchical` | Enable hierarchical staging |
| `--tier-device-expert-gb` | Max GiB for device expert slots |
| `--tier-ram-gb` | Pinned RAM cache GiB (`-1` = auto) |
| `--tier-disk-path` | ExpertStore directory |
| `--tier-policy quality\|balanced` | Live LFRU repin off/on |
| `--tier-repin-tokens N` | Repin interval (balanced) |
| `--tier-pilot` / `--tier-pilot-real` | Router-lookahead prefetch |
| `--tier-io-workers N` | Disk→RAM workers (default 8) |
| `--tier-direct` | Prefer `O_DIRECT` reads |
| `--tier-usage-path` | Learned usage heat-map path |
| `--tier-dense-prefetch` | Also stage dense attention leftovers |
| `--tier-num-slots` | Override slots per MoE layer |
| `--tier-allow-cuda-graphs` | Experimental graphs (default off) |

## Memory planning

At startup vLLM logs a **tier plan** similar to Colibri’s `coli plan`:

```text
Hierarchical expert tier plan:
  policy=quality
  moe_layers=... local_experts=... slots/layer=...
  device_slots=... GiB
  ram_cache=... GiB
  disk_backing=... GiB
  predicted_bottleneck=pcie_or_ram_hits|nvme|none_full_residency
```

Reservation order: dense resident → KV cache → activation scratch → expert
slots. Hierarchical weight offload **cannot** be combined with UVA
`--cpu-offload-gb`. It may coexist with KV CPU offload; leave headroom in
`--tier-ram-gb` for the KV tier.

## How it works

1. After weight load, full expert packs move to pinned host (and optionally
   ExpertStore on NVMe in post-XPU runtime layout).
2. Each MoE layer gets a fixed **device slot pool** of `E_slots` experts.
   `XpuFusedMoe` / modular kernels see a dense pack of size `E_slots`.
3. On every forward, after `select_experts`, the tier manager **batch-unions**
   unique expert ids, ensures they are in slots (RAM hit → DMA, disk miss →
   O_DIRECT/io thread → DMA), and **remaps** `topk_ids` to slot indices.
4. Optional **PILOT** prefetches the next layer’s experts from a routing hint.
5. **Learned pins** (`.vllm_expert_usage`) keep hot experts in RAM across runs;
   `balanced` policy does live LFRU repin.

## Metrics

Prometheus (when enabled):

- `vllm_tier_expert_hits_total{tier=device|ram|disk}`
- `vllm_tier_expert_dma_bytes_total`
- `vllm_tier_expert_stall_seconds_total`
- `vllm_tier_expert_device_hit_rate`

## Limitations (v1)

- Mutual exclusion with **EPLB** (expert-row ownership conflicts)
- Default **eager** mode; `--tier-allow-cuda-graphs` is experimental on XPU
- Expert Parallelism is supported (tiers operate on local experts)
- Lossy router top-p cutting is intentionally not implemented (`quality` policy)

## Related

- Layer-wise PrefetchOffloader: `--offload-group-size` (whole layers, not experts)
- KV offload: [kv_offloading_usage.md](kv_offloading_usage.md)
