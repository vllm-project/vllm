# PR-B — Residency, pinning, auto slots, learned pins

You are implementing **PR-B** of the Colibri-parity hierarchical plan.
**Depends on PR-A** (metrics available for proving hit-rate / stall changes).

## Goal

Make experts **stay hot** when they fit, and size device slots intelligently so
Mixtral-class models do not thrash at `slots=4` by accident.

## Context

- Planner: `vllm/model_executor/offloader/hierarchical/planner.py`
- Manager park/host: `manager.py` (`park_experts_on_host`, `post_init`)
- RAM cache: `ram_cache.py` (LFRU + pin)
- Usage: `usage.py` (`.vllm_expert_usage`)
- Config: `HierarchicalOffloadConfig` in `vllm/config/offload.py` + CLI in
  `vllm/engine/arg_utils.py`

## Requirements

### 1. Auto slot sizing

Improve `compute_slots_per_layer` / `build_tier_plan`:

- If `tier_num_slots > 0`, honor it (clamp to `num_local_experts`).
- If `0`, derive from `tier_device_expert_gb` and free device memory after a
  documented reserve (dense + KV headroom).
- Ensure slots ≥ reasonable batch-union lower bound:
  `min(E, max(top_k, estimated_unique))` with a clear log line.
- Log `full_residency=yes` when `slots >= E`.

### 2. Pinned hot arena vs pageable cold

- Host expert storage must not blindly use `pin_memory=False` for everything.
- Cap **pinned** bytes to `resolve_ram_budget_bytes`.
- Hot experts (from usage + initial fill) live in pinned frames; overflow is
  pageable.
- `_park_module_to_host` must feed the same arenas (no orphan unpinned copies
  that bypass the RAM cache accounting).

### 3. Learned pins that stick

- On `post_init`, seed device slots **and** pinned RAM from
  `ExpertUsageStore.hottest`.
- Wire `notify_tokens` from the serving/offline generate path so
  `tier_policy=balanced` actually calls `repin_hottest` every
  `tier_repin_tokens`.
- Flush usage periodically and on shutdown (verify paths).

### 4. Full-residency fast path

When `slots >= num_local_experts` (and optionally RAM holds the pack):

- Prefer identity remap / minimal ensure churn.
- Still record metrics (PR-A).
- Document as Colibri-like `PIN_GB=all` analogue in
  `docs/features/hierarchical_expert_offload.md`.

### 5. Tests

- Planner unit tests for auto slots and clamping.
- RAM cache: pinned budget respected under overflow.
- Optional: manager residency path doesn’t call disk when unused.

## Non-goals

- Dual-SSD, PILOT rewrite, O_DIRECT alignment, speculation, atlas, graphs.

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Hal (k8s paused): full residency should approach baseline
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 8 --tier-ram-gb 8 --warm 8 --max-tokens 64 \
  --output /tmp/hier_pr_b_residency.json
# Forced staging
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --warm 8 --max-tokens 64 \
  --output /tmp/hier_pr_b_slots4.json
```

Acceptance: with `slots=E`, warm hierarchical decode within ~10% of baseline
tok/s (same hardware). Report both JSON files in the PR.

## PR description must include

- Duplicate-PR check notes, tests/results, AI-assist statement.
- Before/after hit rates from PR-A metrics if available.

## Done when

- Auto slots + pinned hot path + usage seeding + residency fast path landed.
- Docs + tests updated; branch ready for review.
