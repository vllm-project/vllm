# XPU graphs canary — TRITON_ATTN + PIECEWISE

**Branch:** `xpu-graphs-triton-piecewise-canary`  
**Base:** `krisclarkdev/vllm` @ `577e1a932` (fork tip / `hal/vllm-xpu:kris-fork-577e1a932`)  
**Status:** plan only — no production cutover, no DaemonSet default change, no push until requested.

## Summary

Enable a **safe** XPU Graph canary that starts and decodes without capturing FlashAttention kernels that use `sycl_ext_oneapi_work_group_scratch_memory` (that combo crashes SYCL Graph today).

**Target canary knobs (config-first):**

| Knob | Value |
|------|--------|
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `1` |
| `--enforce-eager` | **omit** |
| Attention | `--attention-backend TRITON_ATTN` |
| CUDA/XPU graph mode | `-cc.cudagraph_mode=PIECEWISE` (or `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`) |

**Production Ornith** stays on eager: `--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0` until the canary is proven.

**Finding:** On this tree, **TRITON_ATTN and PIECEWISE are already wired for XPU.** The first canary should be **mostly config / deploy profile**, with at most a small defensive guard in `XPUPlatform.check_and_update_config`. No kernels fork change required for v1.

---

## Current-state findings checklist

Audit @ `577e1a932`:

- [x] **`VLLM_XPU_ENABLE_XPU_GRAPH`** — registered in `vllm/envs.py`; read in `vllm/platforms/xpu.py` `XPUPlatform.check_and_update_config`. Default `0` → forces `cudagraph_mode=NONE`.
- [x] **`supports_xpu_graph()`** — `vllm/utils/torch_utils.py`: requires torch `>= 2.11.0.dev`. If false → graphs forced off.
- [x] **XPU graph wrappers** — `vllm/v1/worker/xpu_model_runner.py` aliases `torch.cuda.CUDAGraph` → `torch.xpu.XPUGraph` when supported; worker is `vllm.v1.worker.xpu_worker.XPUWorker`.
- [x] **`enforce_eager`** — `vllm/config/vllm.py`: if set, forces `cudagraph_mode=NONE` even when graphs env is on. Production path relies on this.
- [x] **Attention backend selection (XPU)** — `XPUPlatform.get_attn_backend_cls`: default is **FLASH_ATTN**; `TRITON_ATTN` is honored when selected; float32 / mm_prefix fall back to Triton.
- [x] **Correct API to select Triton** — **`--attention-backend TRITON_ATTN`** (or `--attention-config.backend=TRITON_ATTN`). There is **no** current first-class `VLLM_ATTENTION_BACKEND` env on this tree for V1; use CLI / config.
- [x] **`cudagraph_mode`** — set via `-cc.cudagraph_mode=...` / `--compilation-config`. Defaults from `-O2` are typically `FULL_AND_PIECEWISE`.
- [x] **FA vs Triton graph capability** — `FlashAttentionBackend`: FA2 ⇒ `UNIFORM_BATCH` (allows **FULL** uniform decode graphs). `TritonAttentionBackend`: `ALWAYS`. Platform **warns** “FLASH_ATTN supports PIECEWISE mode only; use TRITON_ATTN for FULL” but **does not enforce** PIECEWISE when FA + graphs are on.
- [x] **Root cause of prior crash (consistent with code)** — graphs on + default FA + default `FULL_AND_PIECEWISE` → uniform decode takes **FULL** path → FA SYCL kernels with `work_group_scratch_memory` enter XPU Graph → RuntimeError. Rollback to eager worked.
- [x] **PIECEWISE semantics** — attention (opaque op) stays outside the captured piecewise graph; rest of model can capture. Matches canary goal even if someone later mistakenly uses FA (still safer than FULL).
- [x] **Kernels fork** — not required for this canary plan; softcap/fail-closed/MXFP8 MoE work is orthogonal. Confirm Triton attn path does not call FA2 SYCL scratch kernels.
- [x] **Deploy artifacts** — `docker/Dockerfile.xpu` exists in-repo; **no** `deploy/vllm-xpu-daemonset.yaml` in this fork tree (lives elsewhere on hal). Canary manifests belong on **this feature branch only** under `deploy/xpu-graphs-canary/`.
- [x] **Ornith / hybrid** — XPU supports hybrid KV (`support_hybrid_kv_cache`); GDN block-size fix exists. Ornith is **last** on the validation ladder, not first.

**Already works (config-only canary):** enable graphs env + Triton backend + explicit `PIECEWISE` + no enforce-eager.

**Missing / gap (optional small code):** no automatic downgrade to `PIECEWISE` when graphs-on + FLASH_ATTN; a misconfigured canary can still hit the FA-in-FULL crash. Recommend a defensive override in `xpu.py` as Task 2 (not required for a carefully tagged canary profile).

---

## Design

### Canary profile (separate from production Ornith)

```text
Production Ornith DS (unchanged):
  --enforce-eager
  VLLM_XPU_ENABLE_XPU_GRAPH=0
  (default FA OK)

Canary DS / Job (feature-branch manifests only):
  VLLM_XPU_ENABLE_XPU_GRAPH=1
  # do NOT pass --enforce-eager
  --attention-backend TRITON_ATTN
  -cc.cudagraph_mode=PIECEWISE
  # keep single-GPU / ZE_AFFINITY_MASK as today
```

### Why TRITON_ATTN + PIECEWISE (not FA + PIECEWISE alone)

| Combo | Expected |
|-------|----------|
| Graphs + FA + `FULL_AND_PIECEWISE` (prior fail) | Crash on FA scratch-in-graph |
| Graphs + FA + `PIECEWISE` | Likely boots (attn outside graph); still FA risk if mode drifts |
| Graphs + **TRITON_ATTN** + **PIECEWISE** | Intended canary: no FA SYCL scratch in graph; attn graphable but mode keeps attn outside capture |
| Graphs + TRITON_ATTN + `FULL_AND_PIECEWISE` | Allowed by platform docs for “FULL”; **out of scope for v1 canary** (harder correctness surface) |

### Minimal code vs config-only

| Change | Priority | Files / symbols |
|--------|----------|-----------------|
| Canary env + serve flags | **P0** | Deploy/docs on this branch only |
| Optional: if `VLLM_XPU_ENABLE_XPU_GRAPH` and resolved backend is FLASH_ATTN and `cudagraph_mode.has_full_cudagraphs()`, force `CUDAGraphMode.PIECEWISE` + log | **P1** | `vllm/platforms/xpu.py` → `XPUPlatform.check_and_update_config` (may need to run after backend resolution or document “must set PIECEWISE before serve”) |
| Startup log assertion helpers for canary smoke | **P2** | scripts under `deploy/xpu-graphs-canary/` |
| Production DS default flip | **Forbidden** | — |

**Note on P1 timing:** `check_and_update_config` runs before layers pick backends. Safest P1 is: when graphs enabled, **default** `cudagraph_mode` to `PIECEWISE` unless user explicitly set another mode; and document that FA+FULL is unsupported. Prefer not guessing backend inside early config.

### MXFP4 MoE / later Ornith

Keep quantization / MoE paths unchanged. Canary first uses **dense** BF16/FP16 instruct models so graph capture is not entangled with MXFP4 MoE. Only after dense+medium pass, try Ornith with the **same** graphs flags (still no production DS change).

### Rollback

Automatic / ops:

1. Flip canary pod env to `VLLM_XPU_ENABLE_XPU_GRAPH=0` and add `--enforce-eager` (known-good).
2. Or scale canary replicas to 0; leave Ornith DS untouched.
3. Success criteria failed → do not promote flags to Ornith.

---

## Tasks (ordered)

- [ ] **T0** Keep all work on `xpu-graphs-triton-piecewise-canary`; never merge to `main` without explicit ask; never push unless asked.
- [ ] **T1** Confirm runtime torch on `hal/vllm-xpu:kris-fork-577e1a932` reports `supports_xpu_graph()==True` (torch ≥ 2.11.dev).
- [ ] **T2** Add `deploy/xpu-graphs-canary/` on this branch: `README.md`, `env-canary.env.example`, optional `daemonset-canary.yaml.example` (not applied by default) with TRITON_ATTN + PIECEWISE + graphs=1, distinct from Ornith.
- [ ] **T3** (Optional defensive) In `XPUPlatform.check_and_update_config`, when graphs enabled and `cudagraph_mode` still has full graphs, log loudly and optionally clamp to `PIECEWISE` behind a flag e.g. `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1` (default on for safety). Smallest useful guard without surprising FULL Triton experiments.
- [ ] **T4** Run validation ladder (below) on HAL Arc; collect logs + smoke transcripts on this branch under `deploy/xpu-graphs-canary/results/` (gitignored if large).
- [ ] **T5** Document success/fail + rollback in canary README; explicitly “Ornith stays eager”.
- [ ] **T6** Only after ladder green: optional Ornith **shadow** canary (non-default DS), then stop — no production cutover in this effort.

---

## Test plan

### Model ladder

| Step | Model | Why |
|------|--------|-----|
| L1 | Small dense (e.g. `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct`) | Fast fail on graph capture / Triton |
| L2 | Medium dense (e.g. `Qwen/Qwen2.5-7B-Instruct`) | Matches prior XPU wash size |
| L3 | Ornith-1.0-35B MXFP4 | Last; hybrid Mamba/GDN + MoE — only if L1–L2 green |

Do **not** use Ornith as L1. gpt-oss deploy remains out of scope; if anyone experiments, treat decode corruption risk as a hard fail (see Risks).

### Smoke commands (shape)

```bash
# Shared canary env
export VLLM_XPU_ENABLE_XPU_GRAPH=1
# unset / do not set enforce-eager

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --attention-backend TRITON_ATTN \
  -cc.cudagraph_mode=PIECEWISE \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Checks per model:

1. **Startup** — Ready; log contains Triton backend + XPU Graph experimental warning; **no** `work_group_scratch_memory` / SYCL Graph feature error; `cudagraph_mode` not `NONE` unless torch unsupported.
2. **Short decode** — chat completion, `temperature=0`, ~32–64 tokens; non-empty, coherent.
3. **Long decode** — 256–512 tokens; no mid-stream collapse to `!` / empty / garbage loops.
4. **Repetition / determinism** — same prompt twice @ `temperature=0`; outputs match (or differ only by known nondeterminism — fail if wildly different).
5. **Log hygiene** — no NaN warnings; no empty-graph spam; no FA-in-graph crash.
6. **Rollback drill** — set graphs=0 + `--enforce-eager`, confirm recover.

### Success criteria

- L1 and L2: start + short + long + temp=0 repeat pass; no SYCL Graph scratch error.
- Canary profile documented and **not** applied to Ornith production DS.
- Optional: measurable latency/throughput vs eager on L2 (nice-to-have, not gate).

### Fail criteria → rollback

- Any startup RuntimeError involving SYCL Graph / `work_group_scratch_memory`.
- Empty / NaN / `!!!!` decode loops / temp=0 divergence.
- OOM that does not occur under eager at same util (graphs memory tax) — reduce capture sizes or abort canary.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Accidental FA + FULL (default -O2) | Explicit PIECEWISE + TRITON in canary; optional force-PIECEWISE env |
| Triton correctness / perf on XPU weaker than FA | Ladder smokes; keep FA+eager as production |
| Hybrid Ornith (GDN/Mamba) + graphs unknown | Dense-first; Ornith last |
| gpt-oss-style decode corruption under PIECEWISE XPU graphs (upstream reports) | Correctness smokes mandatory; don’t ship on Ready alone |
| Graph memory reduces KV cache | Watch util / max-model-len; compare vs eager |
| Mis-deploy canary flags onto Ornith DS | Separate manifests; branch-only; checklist “Ornith stays eager” |
| Torch &lt; 2.11 → silent NONE | T1 verify `supports_xpu_graph` |

---

## Effort estimate

| Work | Estimate |
|------|----------|
| Canary deploy docs/examples on branch | 0.5–1 d |
| Optional force-PIECEWISE guard | 0.5 d + review |
| L1+L2 HAL smokes + writeup | 1–2 d |
| Ornith shadow (if green) | 1 d |
| **Total to “canary proven / prod still eager”** | **~3–5 d** |

---

## Follow-ups (out of scope for v1)

- Rebuild stack on **oneAPI 2026.0** for SYCL Graph + `work_group_scratch_memory` (FA-in-graph unlock).
- FA SYCL kernel changes for scratch-in-graph.
- Triton `FULL_AND_PIECEWISE` / `FULL` performance canary after PIECEWISE is green.
- Production Ornith cutover runbook (separate change, explicit approval).
- Laguna parsers / gpt-oss deploy.
- Upstreaming defensive XPU graph defaults once proven on fork.

---

## Knobs cheat-sheet (this fork)

```bash
# Enable XPU Graph machinery (required or cudagraph_mode forced NONE)
VLLM_XPU_ENABLE_XPU_GRAPH=1

# Attention — CLI, not VLLM_ATTENTION_BACKEND
--attention-backend TRITON_ATTN

# Graph mode — keep FA out of capture; safest canary
-cc.cudagraph_mode=PIECEWISE

# Production / rollback
VLLM_XPU_ENABLE_XPU_GRAPH=0
--enforce-eager
```

Primary code touchpoints if implementing guards later:

- `vllm/platforms/xpu.py` — `XPUPlatform.check_and_update_config`, `get_attn_backend_cls`
- `vllm/envs.py` — `VLLM_XPU_ENABLE_XPU_GRAPH` (+ optional force-PIECEWISE)
- `vllm/v1/worker/xpu_model_runner.py` — CUDA→XPU graph aliases
- `vllm/v1/attention/backends/triton_attn.py` / `flash_attn.py` — cudagraph support enums
