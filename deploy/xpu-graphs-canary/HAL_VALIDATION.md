# Canary validation notes (HAL)

Date: 2026-07-26

## T1 — supports_xpu_graph

Checked **read-only** inside production Ornith pod `inference/vllm-xpu-q9hv6`
(`hal/vllm-xpu:kris-fork-577e1a932`):

```
torch 2.12.0+xpu
supports_xpu_graph True
```

## T4 — ladder smoke

**Not run.** Ornith DaemonSet holds the only Intel Arc (`gpu.intel.com/xe`).
Plan forbids disrupting production. Next step:

1. Build image from branch `xpu-graphs-triton-piecewise-canary` (includes
   `VLLM_XPU_GRAPH_FORCE_PIECEWISE`).
2. During a maintenance window (scale Ornith to 0) **or** on a second GPU,
   run:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash deploy/xpu-graphs-canary/smoke_canary.sh
MODEL=Qwen/Qwen2.5-7B-Instruct bash deploy/xpu-graphs-canary/smoke_canary.sh
```

3. Leave Ornith on eager (`VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`).
