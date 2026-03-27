# RIY: Runtime Expert Masking for MoE Models

## Overview

RIY ("Pruning on Air") provides runtime expert deactivation and activation
statistics for Mixture-of-Experts models. Experts are masked at the routing
level without modifying the checkpoint or model structure.

## Architecture

```
                        Admin API (/riy/*)
                             |
                    +--------+--------+
                    |    RiyState     |  Global singleton
                    |  - layer_stats  |  Stats: freq + weight_sum per (layer, expert)
                    |  - mask         |  Mask: set of (layer, expert) tuples
                    |  - mask_tensors |  Pre-computed bool tensors per layer
                    +--------+--------+
                             |
               BaseRouter.select_experts()
                             |
            +----------------+----------------+
            |                |                |
     _compute_routing   record_stats    apply_riy_mask
     (existing vLLM)    (if collecting)  (if mask set)
            |                                 |
            v                                 v
       topk_weights, topk_ids          masked_fill + renormalize
```

## Files

| File | Lines | Role |
|------|-------|------|
| `vllm/model_executor/layers/fused_moe/riy.py` | ~230 | Core: state, stats accumulator, mask, renormalization |
| `vllm/entrypoints/serve/riy_api.py` | ~110 | Admin API: 8 REST endpoints |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | +16 | Hook in select_experts(): stats + mask |
| `vllm/model_executor/layers/fused_moe/router/router_factory.py` | +7 | Pass layer_idx through factory |
| `vllm/model_executor/layers/fused_moe/layer.py` | +9 | Layer registration + index extraction from prefix |
| `vllm/config/parallel.py` | +4 | CLI flag `--riy-expert-profile` |
| `vllm/entrypoints/serve/__init__.py` | +6 | Register API router |
| 5x router subclasses | +2 each | Accept layer_idx parameter |

**Total**: ~52 lines changed in existing files, ~340 lines in new files.

## Design Principles

1. **No automatic pruning.** The system provides data. The operator decides.
2. **Raw data export.** Statistics are unfiltered: activation frequency and
   output magnitude per `(layer, expert)`. No scoring logic in the inference system.
3. **Reversibility.** Any mask can be removed or replaced at any time.
   The on-disk model is never modified.
4. **Quantization-agnostic.** A profile is a list of `(layer, expert)` tuples.
   Same profile works on BF16, FP8, INT4.
5. **No re-indexing.** Masked experts keep their index. The router still
   addresses them; their weight is zeroed and remaining weights renormalized.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/riy/stats` | Export raw expert statistics |
| POST | `/riy/stats/start` | Start stats collection |
| POST | `/riy/stats/stop` | Stop stats collection |
| POST | `/riy/stats/reset` | Reset all counters |
| GET | `/riy/mask` | Get current expert mask |
| POST | `/riy/mask` | Set expert mask (JSON body) |
| DELETE | `/riy/mask` | Clear expert mask |
| POST | `/riy/profile/load` | Load mask from profile JSON on disk |

## Profile Format

```json
{
  "version": 1,
  "model": "Qwen3.5-397B-A17B",
  "workload": "municipal German administrative",
  "pruned_experts": [[0, 3], [0, 11], [4, 7], [12, 2]]
}
```

## Usage

### Collect statistics and build a profile

```bash
# Start vLLM with VLLM_RIY_MONITOR=1 (enables stats + HTTP server)
# Stats collection starts automatically on first forward pass
VLLM_RIY_MONITOR=1 vllm serve Qwen/Qwen3.5-397B-A17B --port 8011

# Or start/stop collection manually via API:
curl -X POST http://localhost:8019/riy/stats/start
curl -X POST http://localhost:8019/riy/stats/stop

# Reset counters (clean start, no warm-up carry-over)
curl -X POST http://localhost:8019/riy/stats/reset

# Run your workload...

# Export stats
curl http://localhost:8011/riy/stats > stats.json

# Use offline tool to build profile from stats
# (separate project: github.com/flash7777/riy)
```

### Apply a profile at runtime

```bash
# Set mask (immediate, no restart)
curl -X POST http://localhost:8011/riy/mask \
  -H 'Content-Type: application/json' \
  -d '{"pruned_experts": [[0,3],[0,11],[4,7]]}'

# Or load from file on disk
curl -X POST http://localhost:8011/riy/profile/load \
  -H 'Content-Type: application/json' \
  -d '{"path": "/data/profiles/my_workload.json"}'

# Check quality, then clear if needed
curl -X DELETE http://localhost:8011/riy/mask
```

### Apply a profile at load time

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --riy-expert-profile /data/profiles/my_workload.json
```

## Performance

- **Fast path**: When no mask is active and stats collection is off,
  the hook checks two booleans and returns. Zero overhead on inference.
- **Stats collection**: One `scatter_add_` per layer per forward pass,
  on CPU. Negligible compared to MoE computation.
- **Mask application**: One `masked_fill` + one division per layer.
  Sub-microsecond for typical expert counts.

## Limitations

- **No VRAM savings at runtime.** Masked experts still occupy memory.
  The load-time mode (`--riy-expert-profile`) zeros weights but does
  not skip allocation. True memory savings require skipping tensor
  allocation entirely (planned as follow-up).
- **Tensor Parallel.** The mask must be consistent across all TP ranks.
  Currently the global singleton is per-process, which works for TP
  since all ranks see the same routing decisions.
