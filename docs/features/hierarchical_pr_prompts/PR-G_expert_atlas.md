# PR-G — Expert atlas + affinity-aware pins (optional)

You are implementing **PR-G** of the Colibri-parity hierarchical plan.
**Depends on PR-B** (learned pins). This PR is **optional** and should stay
behind flags.

## Goal

Port Colibri’s **expert atlas** idea: offline probes measure which experts fire
for which topics; at runtime, boost pins for the session’s topic affinity so
cold-start hit rate improves.

## Context

- Usage store: `usage.py` (heat counts only today)
- RAM pins: `ram_cache.py` (`pin`, `repin_hottest`)
- Colibri reference: expert atlas tools / issue discussion on JustVugg/colibri
  (affinity is **measured routing**, not a learned embedding)

## Requirements

### 1. Offline atlas tool

Add a script under `benchmarks/` or `tools/` (prefer existing layout) that:

- Loads a MoE with hierarchical or full weights.
- Runs a probe prompt set (JSON config) with teacher-forcing or short
  generates.
- Records per-(layer, expert) hit counts **tagged by probe topic**.
- Writes a sidecar next to usage, e.g. `.vllm_expert_atlas.json`.

### 2. Runtime affinity pins

- Config: `--tier-atlas-path` and optional `--tier-affinity-topic`.
- On init / session start, if atlas present, boost pin scores for experts
  matching the topic (or auto topic from system prompt keywords — keep v1
  simple: explicit topic id).
- Fallback to pure LFRU/usage if atlas missing.

### 3. Eval

- Held-out topic prompts: affinity pins vs cold LFRU — compare device/RAM hit
  rates (PR-A metrics) and tok/s.

### 4. Docs

- Short section in `docs/features/hierarchical_expert_offload.md` on building
  and using an atlas.
- Clear statement that atlas never changes router outputs—only placement.

## Non-goals

- 3D visualization UI (Colibri Brain/Atlas web).
- Changing the router or model weights.
- Dual-SSD / speculation (unless already present).

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Offline atlas on a tiny MoE or Mixtral smoke probes
```

Acceptance: atlas file round-trips; affinity mode improves hit rate on a
tagged topic vs cold start in a reported bakeoff snippet.

## PR description must include

- Duplicate check, tests/results, AI-assist statement, sample atlas schema.

## Done when

- Offline tool + optional runtime affinity pins mergeable without affecting
  default hierarchical behavior.
