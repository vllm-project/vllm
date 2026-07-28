# PR-D — Dual-NVMe mirror + optional NUMA

You are implementing **PR-D** of the Colibri-parity hierarchical plan.
**Depends on PR-C** (disk PIPE / ExpertStore reader is trustworthy).

## Goal

Raise aggregate disk→RAM bandwidth using a **second identical model copy**, and
optionally place pinned host arenas with **NUMA** awareness — Colibri’s
dual-SSD + `COLI_NUMA` ideas in vLLM.

## Context

- Disk reader: `disk_store.py`, format/manifest: `format.py`
- Config: `HierarchicalOffloadConfig` + `arg_utils.py` CLI
- Metrics from PR-A should grow to include per-volume bytes

## Requirements

### 1. Dual disk mirror

Add config/CLI:

- `--tier-disk-path` (primary, existing)
- `--tier-disk-mirror` (optional second root)
- `--tier-disk-weights a,b` optional bandwidth ratio; else probe at startup

Behavior (match Colibri semantics):

- Validate mirror files (size + safetensors/header identity where applicable).
- Partial mirror OK: missing files stay on primary.
- Deterministic hash(layer_id, expert_id) → primary vs mirror using weights.
- PILOT and demand for the same expert **must** use the same volume.
- Mirror is read-only; usage/KV/sidecars stay on primary.
- On mirror read error: warn once, fall back to primary; do not crash.
- Log a `MIRROR:` stats line (GB served per drive).

### 2. Optional NUMA

- `--tier-numa` / env: when enabled on Linux multi-node hosts, allocate pinned
  expert frames with interleave or bind policy (document exact API used).
- No-op with a clear log if unsupported.

### 3. Tests

- Hash routing stability + weight skew.
- Validation accepts partial mirror.
- Fallback path unit-tested with a fake failing reader.

## Non-goals

- Rewriting ExpertStore format unless required for dual roots.
- Speculation, atlas, graphs.

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Hardware (two NVMe mounts), if available:
# --tier-disk-path /nvme0/expert_store --tier-disk-mirror /nvme1/expert_store
```

Acceptance: synthetic or real dual-path read shows bytes on both volumes; single
volume still works unchanged.

## PR description must include

- Duplicate check, tests/results, AI-assist statement, mirror bandwidth notes.

## Done when

- Mirror + optional NUMA shipped behind flags; default single-disk behavior
  unchanged.
