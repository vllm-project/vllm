# 03 — FlashAttention-in-graph / fuller XPU graph modes (small dense)

**Branch:** `fa-in-graph-dense-validation` (`krisclarkdev/vllm`)
**Base:** fork `main` @ `5cddd6167` (PR #2 = 01 PIECEWISE canary, PR #3 = 02 oneAPI 2026 base image, both merged)
**Image tag:** `hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` (from 02; `torch 2.13.0+xpu`, `intel-sycl-rt 2026.0.0`, `triton-xpu 3.7.2`, kernels `@aa156578`)
**Status:** implemented + validated on `hal` (Arc Pro B70). FA captured in a full XPU Graph, Ready, correct decode, ~+500% single-stream decode vs eager. No production Ornith / DaemonSet cutover.

---

## Summary

02 rebuilt the XPU runtime on oneAPI **2026.0** + `torch 2.13.0+xpu` and its **T7 probe proved** that `vllm_xpu_kernels.flash_attn_varlen_func` now captures and replays inside a `torch.xpu.XPUGraph` **without** the `sycl_ext_oneapi_work_group_scratch_memory` SYCL Graph error. That was the hard blocker for FlashAttention-in-graph.

01 shipped a **default-on safety clamp** (`VLLM_XPU_GRAPH_FORCE_PIECEWISE=1`) in `XPUPlatform.check_and_update_config` that downgrades any full graph mode to `PIECEWISE`, keeping FA kernels **outside** the captured graph. On the old 2025.3 base that clamp was the only thing preventing the crash.

Feature 03 turns FA-in-graph on **safely** for **small dense** models:

- **Enablement (config):** graphs on + FA backend + a full graph mode + opt out of the 01 clamp (`VLLM_XPU_GRAPH_FORCE_PIECEWISE=0`), no `--enforce-eager`.
- **Minimal code:** a runtime capability gate `supports_xpu_fa_in_graph()` plus a **fail-closed** guard so that opting out of PIECEWISE on a runtime that *cannot* capture FA scratch kernels (old base) re-clamps to PIECEWISE and warns loudly, instead of regressing into the `work_group_scratch_memory` crash.

Production Ornith stays eager. This validates dense only — not Ornith MXFP4 MoE / hybrid Mamba (that is `04`).

---

## Design

### Current state on merged `main` (audit)

| Piece | Value | File / symbol |
|-------|-------|---------------|
| Graph master switch | `VLLM_XPU_ENABLE_XPU_GRAPH` (default `0`) → forces `cudagraph_mode=NONE` when off | `vllm/envs.py`, `vllm/platforms/xpu.py::check_and_update_config` |
| Full-graph safety clamp | `VLLM_XPU_GRAPH_FORCE_PIECEWISE` (default `1`) → `has_full_cudagraphs()` modes clamped to `PIECEWISE` | `vllm/platforms/xpu.py::check_and_update_config` |
| torch graph capability | `supports_xpu_graph()` → torch ≥ `2.11.0.dev` | `vllm/utils/torch_utils.py` |
| FA cudagraph support | XPU FA is FA2 ⇒ `AttentionCGSupport.UNIFORM_BATCH` (allows FULL uniform-decode graphs, drops to `FULL_AND_PIECEWISE`) | `vllm/v1/attention/backends/flash_attn.py::_cudagraph_support` |
| Default attention (XPU) | `FLASH_ATTN`; `TRITON_ATTN` honored when selected | `vllm/platforms/xpu.py::get_attn_backend_cls` |
| Stale warning | "FLASH_ATTN supports PIECEWISE mode only; use TRITON_ATTN for FULL" — no longer true on oneAPI 2026 | `check_and_update_config` |

So FA + FULL is **not blocked** by validation anywhere; it is only **clamped** by the default-on 01 guard. Enabling FA-in-graph = opt out of the clamp on a capable runtime.

### Runtime capability signal

02 recorded `torch.version.xpu == 20260000` on the oneAPI 2026.0 base (vs `2025xxxx` on the old 2.12 base). FA SYCL kernels use `work_group_scratch_memory`, which only captures into a SYCL Graph on oneAPI **2026.0+**. So:

```python
def supports_xpu_fa_in_graph() -> bool:
    # FlashAttention SYCL kernels use sycl_ext_oneapi_work_group_scratch_memory,
    # which can only be captured into a SYCL Graph on oneAPI 2026.0+ runtimes.
    if not supports_xpu_graph():
        return False
    xpu_ver = getattr(torch.version, "xpu", None)
    try:
        return xpu_ver is not None and int(xpu_ver) >= 20260000
    except (TypeError, ValueError):
        return False
```

### Guard logic (in `check_and_update_config`, graphs-enabled branch)

For a full graph mode (`cudagraph_mode.has_full_cudagraphs()`):

1. `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1` (**default**) → clamp to `PIECEWISE` (unchanged 01 behavior; FA stays out of capture).
2. `VLLM_XPU_GRAPH_FORCE_PIECEWISE=0` **and** `not supports_xpu_fa_in_graph()` → **fail-closed**: re-clamp to `PIECEWISE` and emit a loud warning that the runtime lacks scratch-in-graph (prevents the known `work_group_scratch_memory` crash if someone runs the 02-era flags on a wrong/old base).
3. `VLLM_XPU_GRAPH_FORCE_PIECEWISE=0` **and** `supports_xpu_fa_in_graph()` → **allow FA-in-graph**; log that full XPU Graph capture including FlashAttention is enabled.

Also update the stale warning (2) to reflect that FA full-graph capture is supported on oneAPI 2026+ and gated by `VLLM_XPU_GRAPH_FORCE_PIECEWISE`.

**No default change:** the safety clamp stays default-on. FA-in-graph is strictly opt-in. Production Ornith (which does not set these flags and runs `--enforce-eager`) is unaffected.

### Files / symbols to change

| File | Change |
|------|--------|
| `vllm/utils/torch_utils.py` | Add `supports_xpu_fa_in_graph()` |
| `vllm/platforms/xpu.py` | `check_and_update_config`: capability-aware clamp (3-way) + updated warning text; import helper |
| `deploy/xpu-fa-graph-canary/` | `README.md`, `env-canary.env.example`, `smoke_fa_graph.sh`, `ab_fa_graph.sh` (branch-only; separate from Ornith) |
| `docs/xpu/FA_IN_GRAPH_DENSE_VALIDATION_PLAN.md` | this plan + recorded results |

**Kernels:** verify only. FA-in-graph capture uses `vllm_xpu_kernels @ aa156578` unchanged (02 T7 proved capture works at that pin). No kernels feature branch expected; document verification.

### Canary env (small dense)

```bash
# small dense model, e.g. Qwen/Qwen2.5-0.5B-Instruct (matches 02 A/B) or 1.5B
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_GRAPH_FORCE_PIECEWISE=0     # opt out of 01 clamp -> allow FA in graph
export ZE_AFFINITY_MASK=0                    # single GPU
# do NOT pass --enforce-eager

vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8020 \
  --attention-backend FLASH_ATTN \
  -cc.cudagraph_mode=FULL \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16
```

Rollback path (known-good): `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1` (→ 01 PIECEWISE) or `VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`.

---

## Test plan

All off production Ornith; pin to a free GPU via `ZE_AFFINITY_MASK`.

1. **Unit** — `supports_xpu_fa_in_graph()` and the 3-way clamp: parametrized test asserting FULL→FULL when capable + opt-out, FULL→PIECEWISE when clamp on, FULL→PIECEWISE + warn when opt-out + not capable. CPU-only (monkeypatch `torch.version.xpu` / helpers).
2. **Startup** — FA-in-graph canary env: Ready; log shows FlashAttention backend + `cudagraph_mode: FULL`/`FULL_AND_PIECEWISE` + FA-in-graph enabled line; **no** `work_group_scratch_memory` / SYCL Graph error; not `NONE`.
3. **Short decode** — `temperature=0`, ~32 tokens; `2+2 → 4`; coherent.
4. **Long decode** — 256–512 tokens; no `!{4,}` loop, no empty/NaN/garbage.
5. **temp=0 repeat** — same prompt twice; identical text.
6. **Eager baseline compare** — same model `--enforce-eager`; FA-in-graph output text matches eager greedy output.
7. **Fail-closed drill** — set `VLLM_XPU_GRAPH_FORCE_PIECEWISE=0` behavior confirmed to clamp on a non-2026 signal (unit-level; do not need the old image live).

### Success criteria

Ready + correct decode (matches eager greedy) on the small dense model with FA captured in a full XPU Graph, no scratch-in-graph error. A/B (eager vs FA-in-graph) recorded.

### Fail → rollback

Any `work_group_scratch_memory` / SYCL Graph error, NaN, `!` loop, empty output, or temp=0 divergence from eager → auto-document, roll the canary back to PIECEWISE (01) or eager. Do not promote to Ornith.

---

## A/B

Arms on the 02 image, small dense, greedy, identical prompts:

- **A (baseline):** `--enforce-eager`, graphs off.
- **B (feature):** graphs on + FA + `cudagraph_mode=FULL` + `FORCE_PIECEWISE=0` (FA-in-graph).
- Optional **C:** 01 PIECEWISE + TRITON_ATTN (context vs 03).

Collect TTFT + decode tok/s (single-stream) and/or `vllm bench serve` median output tok/s; assert identical output text A vs B. Record JSON + markdown under `deploy/xpu-fa-graph-canary/results/`.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Scratch-in-graph still incomplete in practice despite version bump | Startup + capture smoke gates on `work_group_scratch_memory`; fail-closed guard re-clamps if capability signal absent |
| Correctness bug only under long decode | Mandatory long-decode + temp=0 + eager-compare smokes; don't ship on Ready alone |
| Scope creep into kernels | Verify `@aa156578` capture only; no kernels change unless capture proven broken |
| Confusing FA-in-graph success with Ornith readiness | Dense-only here; Ornith MXFP4/hybrid is `04`; production stays eager |
| Default-behavior regression | Safety clamp stays default-on; FA-in-graph strictly opt-in |
| Graph memory tax → OOM/less KV | Compare util vs eager; reduce capture sizes / max-model-len if needed |

---

## Ordered tasks + effort

| Task | Est. |
|------|------|
| **T0** Plan (this doc) | done |
| **T1** `supports_xpu_fa_in_graph()` helper | 0.25 d |
| **T2** Capability-aware clamp + warning in `xpu.py` | 0.25 d |
| **T3** Unit test for helper + clamp | 0.25 d |
| **T4** Canary deploy assets (README/env/smoke/ab) | 0.25 d |
| **T5** hal: FA-in-graph correctness smokes (startup/short/long/temp0/eager-compare) | 0.5 d |
| **T6** hal: A/B eager vs FA-in-graph, record results | 0.5 d |
| **T7** Lint + PR + merge to fork main | 0.25 d |
| **Total** | **~2–2.5 d** |

All tasks complete (T0–T7).

---

## Results (hal, Arc Pro B70)

Image `hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` + branch overlay
(`vllm/envs.py`, `vllm/platforms/xpu.py`, `vllm/utils/torch_utils.py`).
Runtime: `torch 2.13.0+xpu`, `torch.version.xpu=20260000`, `intel-sycl-rt 2026.0.0`,
`triton-xpu 3.7.2`, kernels `0.1.12.dev33+gaa15657` (`@aa156578`, unchanged).
Model: `Qwen/Qwen2.5-0.5B-Instruct` bf16, greedy, single stream.

| Arm | TTFT ms mean | Decode tok/s mean |
|-----|--------------|-------------------|
| A: eager (`--enforce-eager`, graphs off) | 36.1 | 67.0 |
| B: FA-in-graph (`FLASH_ATTN` + `cudagraph_mode=FULL` + `FORCE_PIECEWISE=0`) | 22.6 | 401.6 |

**Decode +499.5%, TTFT −37.6%** (matches 02's FULL-graph magnitude).

- Startup Ready; log: `FlashAttention-in-graph enabled: capturing full XPU Graph
  (mode FULL) including FlashAttention on oneAPI 2026.0+ runtime.`
- Captured 51 PIECEWISE + 35 FULL graphs; `FULL → FULL_AND_PIECEWISE` (FA2
  `UNIFORM_BATCH` downgrade). **No `work_group_scratch_memory` error.**
- `VLLM_XPU_GRAPH_FORCE_PIECEWISE` registered (no unknown-env warning).
- Correctness: per-arm temp=0 byte-identical; FA-in-graph greedy matches eager
  for the first 681/≈768 chars, diverging only at the last ~1–2 tokens (bf16
  near-tie flip), both coherent; no NaN / `!` loops / garbage.
- Unit: `tests/utils_/test_torch_utils.py::test_supports_xpu_fa_in_graph` (6/6).

Artifacts: `deploy/xpu-fa-graph-canary/results/AB_COMPARE_20260727T151703Z.{json,md}`.

---

## Follow-ups → `04` Ornith canary

- `04`: same graph flags on Ornith-1.0-35B MXFP4 MoE / hybrid Mamba (dense-first proven here). Still no production DS cutover.
- `05`: production Ornith cutover runbook (separate, explicit approval).
- Optional: upstream a capability-aware XPU graph default once proven on the fork.

---

## References

- 02 build notes `docs/xpu/ONEAPI_2026_BUILD_NOTES.md` — T7 FA-in-graph probe OK; FULL graph serve OK; `torch.version.xpu=20260000`.
- 01 canary plan `docs/xpu/XPU_GRAPHS_TRITON_PIECEWISE_CANARY_PLAN.md` — `VLLM_XPU_GRAPH_FORCE_PIECEWISE` guard, PIECEWISE rollback.
- `vllm/v1/attention/backends/flash_attn.py` — FA2 `UNIFORM_BATCH` cudagraph support.
