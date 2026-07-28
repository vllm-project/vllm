# PR-A — Measurement spine + bakeoff harness

You are implementing **PR-A** of the Colibri-parity hierarchical expert offload
speed plan for vLLM (XPU-first). Do **not** implement later PRs in this change.

## Goal

Make hierarchical performance **measurable** so later PRs can prove wins.

## Context

- Code lives under `vllm/model_executor/offloader/hierarchical/`
  (`manager.py`, `metrics.py`, `device_slots.py`, …).
- `TierStats` / Prometheus helpers exist but e2e often reported empty
  `tier_stats`.
- Bakeoff stub: `benchmarks/hierarchical_tier_bakeoff.py`.
- Eval notes: `docs/features/hierarchical_expert_offload_eval.md`.

## Requirements

### 1. Per-forward / per-step counters

Extend `TierStats` (and Prometheus) so after a generate you can see:

- `device_hits`, `device_misses`
- `ram_hits`, `ram_misses`
- `disk_hits`, `disk_misses` (0 if no disk store)
- `h2d_bytes`, `h2d_stall_ns` (time blocked waiting on copy stream)
- `disk_bytes`, `disk_wait_ns`
- `unique_experts_sum` / optional histogram
- `ensure_calls`

Wire increments in `ExpertTierManager._ensure_layer` / `ensure_and_remap`
(and disk reader). Ensure `get_tier_manager().stats.snapshot()` is non-empty
after at least one MoE forward with hierarchical enabled.

### 2. Bakeoff harness

Upgrade `benchmarks/hierarchical_tier_bakeoff.py` to:

- Load once with hierarchical, warm `W` steps (discard), then measure decode
  tok/s (and optional prefill) over `N` prompts.
- Optionally run baseline (no hierarchical) in a **separate** process/invocation
  and write a comparison JSON.
- Fields: `tok_s_warm`, `ttft_proxy`, hit rates from `TierStats`, config dump,
  optional `--colibri-tok-s` reference.
- Default model path suitable for hal (document Mixtral-8x7B AWQ and Ornith).

### 3. Unit tests

- Extend `tests/model_executor/offloader/test_hierarchical_offload.py` to assert
  counters move on a fake ensure path (CPU/mock OK).

### 4. Docs

- Update `docs/features/hierarchical_expert_offload_eval.md` with the new JSON
  schema and commands.

## Non-goals

- No dual-SSD, PILOT fixes, speculation, atlas, or graph work.
- No change to AWQ conversion or slot remapping semantics.

## Test plan

```bash
uv pip install -r requirements/lint.txt  # if needed
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# On hal (k8s paused), smoke bakeoff:
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --warm 4 --max-tokens 32 \
  --output /tmp/hier_pr_a_bakeoff.json
```

## PR description must include

- Why this is not a duplicate of an open PR.
- Test commands + results.
- Statement that AI assistance was used (if applicable).
- Note: metrics-only; no claimed tok/s regression/improvement yet.

## Done when

- Warm hierarchical generate yields non-empty `tier_stats` / bakeoff JSON.
- Unit tests pass.
- Commit on a focused branch; do not push unless asked.
