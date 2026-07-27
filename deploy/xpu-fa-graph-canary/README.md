# XPU FlashAttention-in-graph canary (FA + FULL)

Feature branch: `fa-in-graph-dense-validation`
Full plan: `../../docs/xpu/FA_IN_GRAPH_DENSE_VALIDATION_PLAN.md`
**DO NOT** apply as the production Ornith DaemonSet.

## Goal

Capture **FlashAttention inside a full XPU Graph** on a small dense model, now
that the oneAPI 2026.0 runtime (feature 02) supports `work_group_scratch_memory`
under the SYCL Graph extension. This is the opposite of the 01 canary, which
kept FA *outside* the capture via PIECEWISE.

## Ornith stays eager (production)

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=0
vllm serve ... --enforce-eager ...
```

Never copy canary flags onto the live Ornith DaemonSet without an explicit
cutover decision.

## FA-in-graph canary profile

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_GRAPH_FORCE_PIECEWISE=0   # opt out of 01 clamp -> FA in graph
export ZE_AFFINITY_MASK=0                  # single GPU
# do NOT pass --enforce-eager

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8020 \
  --attention-backend FLASH_ATTN \
  -cc.cudagraph_mode=FULL \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16
```

`VLLM_XPU_GRAPH_FORCE_PIECEWISE=0` on a runtime that cannot capture FA scratch
kernels (pre-2026 base) is **fail-closed**: the platform re-clamps to PIECEWISE
and logs why, instead of crashing with the `work_group_scratch_memory` error.

## Success / fail / rollback

**Success (small dense):**

- Server Ready; log shows FlashAttention backend + `cudagraph_mode: FULL`
  (or `FULL_AND_PIECEWISE`) + `FlashAttention-in-graph enabled` line
- No `work_group_scratch_memory` / SYCL Graph feature error
- Short + long decode coherent; `temperature=0` double-prompt matches
- FA-in-graph greedy output text matches the eager baseline
- No NaN / empty-graph spam / `!!!!` loops

**Fail → rollback canary only (leave Ornith alone):**

1. `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1` (falls back to 01 PIECEWISE), or
2. `VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`

## Scripts

```bash
# Correctness smoke (startup/short/long/temp=0 repeat):
MODEL=/path/to/Qwen2.5-0.5B-Instruct bash deploy/xpu-fa-graph-canary/smoke_fa_graph.sh

# A/B: eager baseline vs FA-in-graph FULL (asserts identical greedy text):
MODEL=/path/to/Qwen2.5-0.5B-Instruct bash deploy/xpu-fa-graph-canary/ab_fa_graph.sh
```

Results under `results/` (large logs gitignored; keep `AB_COMPARE_*`,
`*_meta_*`, `*_smoke_*`).

## Ladder

| Step | Model | Note |
|------|-------|------|
| L1 | `Qwen/Qwen2.5-0.5B-Instruct` | fast fail; matches 02 A/B |
| L2 | `Qwen/Qwen2.5-1.5B-Instruct` | dense confirm |
| L3 | Ornith MXFP4 MoE | **feature 04 only**, not here |
