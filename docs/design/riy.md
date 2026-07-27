# RIY: Profile-driven MoE expert pruning

RIY applies a per-layer expert profile when a model is loaded. It composes the
profile with vLLM's expert placement, prevents pruned experts from being selected
by the router, and avoids allocating their expert weights.

## Load-time pruning

Pass a profile with `--riy-expert-profile`:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --riy-expert-profile /data/profiles/workload.json
```

The equivalent `RIY_EXPERT_PROFILE` environment variable is supported for
compatibility. The CLI option takes precedence.

A profile contains global expert IDs for each MoE layer:

```json
{
  "version": 1,
  "model": "Qwen3-30B-A3B",
  "pruned_experts": [[0, 3], [0, 11], [4, 7]]
}
```

For each layer, RIY:

1. Builds a `-inf` router-logit mask so top-k cannot select pruned experts.
2. Intersects the retained experts with the rank's linear EP placement.
3. Compacts the global-to-local expert map.
4. Reduces `local_num_experts`, so pruned expert weights are not allocated.

A profile that keeps fewer experts than the model's `top_k` is rejected.
Invalid and out-of-range profile entries are also rejected.

Profile pruning supports tensor parallelism and linear expert placement. It
currently rejects EPLB, round-robin expert placement, and a profile that removes
all experts assigned to an EP rank.

## Optional monitoring

Set `VLLM_RIY_MONITOR=1` to collect per-layer expert frequency and routing-weight
statistics. Monitoring also starts a small control server in the EngineCore
process on `127.0.0.1:8019`.

```bash
VLLM_RIY_MONITOR=1 vllm serve Qwen/Qwen3-30B-A3B
curl http://127.0.0.1:8019/riy/stats
```

The server can be bound to another trusted interface with `VLLM_RIY_HOST`.
It has no authentication or transport encryption and must not be exposed to an
untrusted network.

Monitoring state is process-local and is not yet aggregated across TP or EP
workers. Runtime mask updates are intentionally not exposed; changing the
allocated expert set requires a restart with a new load-time profile.

## Implementation

- `vllm/model_executor/layers/fused_moe/riy.py` loads profiles and owns optional
  monitoring state.
- `vllm/model_executor/layers/fused_moe/expert_map_manager.py` composes profile
  filtering with expert placement.
- `vllm/model_executor/layers/fused_moe/layer.py` wires the profile, expert map,
  and router mask into each MoE layer.
- `vllm/model_executor/layers/fused_moe/router/base_router.py` applies the logit
  mask before top-k and records optional monitoring statistics.
