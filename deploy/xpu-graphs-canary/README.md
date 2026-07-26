# XPU graphs canary deploy stubs
#
# Feature branch: xpu-graphs-triton-piecewise-canary
# DO NOT apply as the production Ornith DaemonSet.
#
# Full plan: ../../docs/xpu/XPU_GRAPHS_TRITON_PIECEWISE_CANARY_PLAN.md

## Goal

Safe XPU Graph canary: `TRITON_ATTN` + `PIECEWISE` so FlashAttention SYCL
kernels that use `work_group_scratch_memory` stay **outside** the captured
graph.

## Ornith stays eager (production)

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=0
vllm serve ... --enforce-eager ...
```

Never copy canary flags into the live Ornith DaemonSet without an explicit
cutover decision.

## Canary profile

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_GRAPH_FORCE_PIECEWISE=1   # default; clamps FULL* → PIECEWISE
export ZE_AFFINITY_MASK=0
# do NOT pass --enforce-eager

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --attention-backend TRITON_ATTN \
  -cc.cudagraph_mode=PIECEWISE \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

See `env-canary.env.example` and `daemonset-canary.yaml.example`.

## Success / fail / rollback

**Success (L1 then L2 dense):**

- Server Ready; logs show Triton backend + XPU Graph experimental warning
- No `work_group_scratch_memory` / SYCL Graph feature errors
- Short + long decode coherent; `temperature=0` double-prompt matches
- No NaN / empty-graph spam / `!!!!` loops

**Fail → rollback canary only (leave Ornith alone):**

1. Set `VLLM_XPU_ENABLE_XPU_GRAPH=0` and add `--enforce-eager`
2. Or scale the canary workload to 0 replicas

## Smoke script

```bash
# On HAL / XPU host with vLLM installed:
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash deploy/xpu-graphs-canary/smoke_canary.sh
```

Results (optional) under `deploy/xpu-graphs-canary/results/` (gitignored).

## Ladder

| Step | Model |
|------|--------|
| L1 | `Qwen/Qwen2.5-1.5B-Instruct` |
| L2 | `Qwen/Qwen2.5-7B-Instruct` |
| L3 | Ornith (shadow only; not production DS) |
