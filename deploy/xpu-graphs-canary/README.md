# XPU graphs canary (deploy stubs)

Feature branch only — **do not** apply as the production Ornith DaemonSet.

See the full plan:

[`docs/xpu/XPU_GRAPHS_TRITON_PIECEWISE_CANARY_PLAN.md`](../../docs/xpu/XPU_GRAPHS_TRITON_PIECEWISE_CANARY_PLAN.md)

## Intended canary flags

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=1
# do not pass --enforce-eager

vllm serve "$MODEL" \
  --attention-backend TRITON_ATTN \
  -cc.cudagraph_mode=PIECEWISE \
  ...
```

## Production Ornith (unchanged)

```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=0
vllm serve ... --enforce-eager ...
```
