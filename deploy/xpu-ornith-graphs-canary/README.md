# Ornith XPU graphs canary (MXFP4 MoE + hybrid GDN/Mamba)

Feature branch: `ornith-xpu-graphs-canary`
Full plan + results: `../../docs/xpu/ORNITH_XPU_GRAPHS_CANARY_PLAN.md`
Image: `hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` (02) + this branch as
an overlay (`vllm/_custom_ops.py`, `vllm/envs.py`, `vllm/platforms/xpu.py`,
`vllm/utils/torch_utils.py`).

**DO NOT** apply any of this to the production `vllm-xpu` DaemonSet. The
production profile stays eager (`--enforce-eager`,
`VLLM_XPU_ENABLE_XPU_GRAPH=0`) until 05 deliberately cuts over.

## Arms

| Arm | Mode | Env | Extra serve args |
|-----|------|-----|------------------|
| A | eager baseline (mirrors prod) | `VLLM_XPU_ENABLE_XPU_GRAPH=0` | `--enforce-eager` |
| B | PIECEWISE (01) | `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1` | `--attention-backend FLASH_ATTN -cc.cudagraph_mode=PIECEWISE` |
| C | FA-in-graph FULL (03) | `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_XPU_GRAPH_FORCE_PIECEWISE=0` | `--attention-backend FLASH_ATTN -cc.cudagraph_mode=FULL` |

All arms share the canary serve profile (port **8021**, 32k context, fp8 KV,
`--max-num-batched-tokens 4096` for Mamba page alignment, util 0.85,
`--max-num-seqs 2`). See `env-canary.env.example`.

## Run

```bash
# Single arm, full Ornith smokes (S1-S6) + perf bench:
ARM=A bash deploy/xpu-ornith-graphs-canary/smoke_ornith.sh

# Whole ladder A -> B -> C with stop-on-fail, eager re-verify on graph-arm
# failure, and A/B/C comparison report:
bash deploy/xpu-ornith-graphs-canary/ab_ornith.sh
```

Results land under `results/` (large serve logs gitignored; keep
`AB_COMPARE_*`, `outputs_*`, `arm_*`, `*_meta_*`).

## Smokes (per arm; Ready is never the gate)

- S1 startup + log hygiene (no `work_group_scratch_memory` / SYCL Graph
  error; arm C must log `FlashAttention-in-graph enabled`)
- S2 short greedy decode (`2+2`)
- S3 long decode 512 tokens (no `!!!!` loops / NaN / collapse)
- S4 temp=0 repeat determinism (byte-identical)
- S5 MoE routing stress: code / math / prose / multilingual prompts
- S6 GDN/Mamba state stress: facts planted before ~2k tokens of filler,
  recall must reproduce the planted secret exactly
- S7 (in `ab_ornith.sh`) cross-arm output agreement vs the eager arm

## Rollback / do-not-break-Arc

- Every script traps exit and kills its server; a failed arm cannot keep the
  GPU. Manual: `docker rm -f ornith-canary` (when containerized) or kill the
  `vllm serve` PID printed by the script.
- If a graph arm fails, `ab_ornith.sh` automatically re-runs a short **eager**
  smoke to prove the node still serves correctly in the known-good config,
  then exits non-zero. Eager is the only state the node may be left serving.
- Config rollback: arm C -> B is `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1`; any
  graph arm -> eager is `VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`.
- The production DaemonSet is not touched by anything here. Its pause/unpause
  pattern lives in the ops repo (`RESUME-ORNITH-MXFP4.md`).

## DaemonSet draft

`daemonset-canary.yaml.example` is a **non-applied** draft for the ops repo
(`intel_vllm_triton`) showing how a k8s-side canary would be isolated (own
name, port 8021, opt-in node label `hal.local/vllm-xpu-canary=true` that no
node carries by default). Applying it is out of scope for this feature.
