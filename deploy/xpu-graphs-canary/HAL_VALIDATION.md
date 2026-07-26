# Canary validation notes (HAL)

Date: 2026-07-26

## T1 — supports_xpu_graph

Checked **read-only** inside production Ornith pod
(`hal/vllm-xpu:kris-fork-577e1a932`):

```
torch 2.12.0+xpu
supports_xpu_graph True
```

## A/B — TRITON_ATTN + PIECEWISE vs eager

Stamp: `20260726T145246Z`  
Model: `Qwen/Qwen2.5-Coder-7B-Instruct`  
Build: `docker.io/hal/vllm-xpu:kris-fork-577e1a932` + hostPath overlay of this branch  
Bench: `vllm bench serve` (warmup + 3 timed runs; failed=0)

| Phase | Config | median output tok/s | smoke |
|-------|--------|---------------------:|-------|
| **before** | graphs=0 + `--enforce-eager` + TRITON_ATTN | **142.55** | PASS |
| **after** | graphs=1 + TRITON_ATTN + `-cc.cudagraph_mode=PIECEWISE` | **143.4** | PASS |

**Delta:** +0.60% (essentially flat). After phase captured PIECEWISE graphs (4/4); no FA `work_group_scratch_memory` crash. Ornith restored to eager afterward.

Full write-up: [`results/AB_COMPARE_20260726T145246Z.md`](results/AB_COMPARE_20260726T145246Z.md).

## Production note

Leave Ornith on eager (`VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`) until a clearer throughput win or longer soak on target models.
