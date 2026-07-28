# Hierarchical expert offload — PR prompts

Self-contained implementation prompts for Colibri-parity speed work
(excluding a pure-C engine rewrite). Run them in order; each prompt assumes
prior PRs are merged (or available on the branch).

| File | PR | Scope |
|------|-----|--------|
| [PR-A_metrics_bakeoff.md](PR-A_metrics_bakeoff.md) | A | Measurement spine + bakeoff |
| [PR-B_residency_pinning.md](PR-B_residency_pinning.md) | B | Auto slots, pinned hot arena, learned pins, full residency |
| [PR-C_async_pilot_odirect.md](PR-C_async_pilot_odirect.md) | C | Async ensure, PILOT, O_DIRECT/PIPE |
| [PR-D_dual_ssd_numa.md](PR-D_dual_ssd_numa.md) | D | Dual-NVMe mirror + optional NUMA |
| [PR-E_device_pipeline_graphs.md](PR-E_device_pipeline_graphs.md) | E | On-device activations, slot stability, graphs |
| [PR-F_speculation_coexistence.md](PR-F_speculation_coexistence.md) | F | Spec decode coexistence with hierarchical |
| [PR-G_expert_atlas.md](PR-G_expert_atlas.md) | G | Offline expert atlas + affinity pins |

## Hard rules (every PR)

- Placement only affects **speed** — never precision or router semantics.
- Follow `AGENTS.md`: `uv` / `.venv`, no duplicate busywork PRs, AI-assist
  disclosure in the PR body, test commands + results.
- Prefer extending `vllm/model_executor/offloader/hierarchical/` over new
  parallel systems.
- Do not edit the Colibri plan file under `.cursor/plans/` unless asked.
- Pause production `inference/vllm-xpu` on hal before Arc e2e; restore after.

## Suggested branch naming

```text
feat/hier-pr-a-metrics
feat/hier-pr-b-residency
...
```
