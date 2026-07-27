# 04 — Ornith MXFP4 MoE + hybrid Mamba/GDN XPU graphs canary

**Branch:** `ornith-xpu-graphs-canary` (`krisclarkdev/vllm`)
**Base:** fork `main` @ `52a58f887` (01 PIECEWISE canary, 02 oneAPI 2026 image,
03 FA-in-graph dense validation — all merged)
**Image tag:** `hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` (02; `torch
2.13.0+xpu`, `torch.version.xpu=20260000`, `intel-sycl-rt 2026.0.0`,
`triton-xpu 3.7.2`, kernels `@aa156578`)
**Graph modes under canary:** PIECEWISE (01 clamp) **and** FA-in-graph FULL
(03 opt-out) — ladder, eager baseline first
**Status:** implemented + validated on `hal` (Arc Pro B70). All three arms
(eager / PIECEWISE / FA-in-graph FULL) passed S1–S7. Graphs give ~+420–432%
single-stream decode with no corruption. **Recommendation: go for 05 with arm
C (FA-in-graph, resolves to `FULL_AND_PIECEWISE`).**
**Production:** Ornith DaemonSet left **paused / eager**, untouched. No cutover
here (that is 05).

---

## Summary

Run `Ornith-1.0-35B-MXFP4` (compressed-tensors MXFP4 MoE + hybrid GDN/Mamba
linear attention) under XPU Graphs on `hal`, as an isolated canary, and decide
go/no-go for the 05 production cutover.

03 proved on this image that FlashAttention SYCL kernels capture and replay
inside a full `torch.xpu.XPUGraph` without the
`sycl_ext_oneapi_work_group_scratch_memory` error, and that dense FA-in-graph
decode is ~5–6× eager. That validated **dense only**. Ornith adds three new
surfaces to graph capture:

1. **GDN/Mamba linear-attention layers** — `GDNAttentionMetadataBuilder`
   declares `AttentionCGSupport.UNIFORM_BATCH` (same class as XPU FA2), so
   uniform-decode FULL capture is architecturally supported; never exercised
   on XPU.
2. **MXFP4 MoE kernels** (`_moe_C` topk + fused experts) inside the captured
   region — in **both** PIECEWISE and FULL modes (only attention ops split
   piecewise graphs; MoE is always captured). This is the classic silent-
   corruption surface (upstream gpt-oss-style decode corruption reports), so
   eager-compare smokes gate the result, not Ready.
3. **fp8 KV cache + Mamba page alignment** (`--max-num-batched-tokens 4096`,
   GDN block-size 64 fix) interacting with graph replay.

### Required code fix found during audit

`vllm/_custom_ops.py` still special-cases XPU to **omit `is_padding`** on
`_moe_C.topk_softmax` / `_moe_C.topk_softplus_sqrt` ("TODO: remove after
vllm-xpu-kernels supports is_padding"). Fork kernels `@aa156578` (in the 02
image) **require** the argument — production worked around this with a
runtime ConfigMap patch (`patch_topk_is_padding.py`) that rewrites the
installed file. Without a fix, Ornith cannot even boot eager on this tree.
This branch removes the stale XPU special-case so the tree matches the pinned
kernels and the runtime patch can be retired at 05 cutover.

No other vLLM code change is expected: graphs enablement is config-only via
the 01/03 knobs (`VLLM_XPU_ENABLE_XPU_GRAPH`, `VLLM_XPU_GRAPH_FORCE_PIECEWISE`,
`-cc.cudagraph_mode`). The 03 fail-closed clamp already protects against
running FA-in-graph on an incapable runtime.

---

## Mode choice (from 03)

| Arm | Mode | Flags | Why |
|-----|------|-------|-----|
| A (baseline) | eager | `VLLM_XPU_ENABLE_XPU_GRAPH=0`, `--enforce-eager` | Mirrors production Ornith config; correctness reference + perf baseline |
| B | PIECEWISE | graphs=1, `FORCE_PIECEWISE=1` (default), `-cc.cudagraph_mode=PIECEWISE`, FLASH_ATTN | Lowest-risk graph arm (01-proven shape): FA + GDN stay outside capture; MoE/MLP captured |
| C | FA-in-graph FULL | graphs=1, `FORCE_PIECEWISE=0`, `-cc.cudagraph_mode=FULL`, FLASH_ATTN | 03-proven mode; FA2+GDN `UNIFORM_BATCH` ⇒ resolves to `FULL_AND_PIECEWISE` (uniform-decode FULL graphs). The perf prize (~6× dense decode) |

Ladder: **A → B → C**, stop-on-fail. B and C are independent results: C may
fail while B passes (B would then be the 05 candidate); if both fail, 05 is
no-go and production stays eager.

Ornith full-attention layers use FLASH_ATTN (XPU default; fp8 KV supported).
TRITON_ATTN fallback arm is optional and only if C fails in an FA-specific way.

### Canary serve profile (all arms identical except graph flags)

```bash
vllm serve /models/Ornith-1.0-35B-MXFP4 \
  --host 0.0.0.0 --port 8021 \
  --served-model-name ornith-canary \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":0}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Deviations from production (deliberate, canary-only): port **8021** (prod
8004), `--max-model-len 32768` (not 131072 — leaves headroom for the graph
memory tax; full-context validation belongs to 05), util 0.85 (prod 0.9), no
prefix caching / tool parsers (orthogonal to graphs; reduce variables). Mamba
align keeps `--max-num-batched-tokens 4096` ≥ block 2112 exactly as prod.

---

## Isolation / do-not-break-live-Arc rules

Current node state (verified before planning): the production `vllm-xpu`
DaemonSet is **paused** (nodeSelector `hal.local/vllm-xpu-paused=true`,
Desired 0) since the 02 build window, and its old image
(`kris-fork-577e1a932`) was pruned from hal. **There is no live Ornith
serving today**; the Arc GPU is free. Restoring production is 05's job.

- Canary runs as a **standalone `docker run`** on hal (03 method: 02 image +
  branch overlay bind-mounted over the installed vLLM). It never touches the
  DaemonSet, its ConfigMaps, LiteLLM routes, or port 8004.
- The k8s path is untouched: no `kubectl apply`, no label changes, no image
  retags. A draft canary DaemonSet manifest is provided for later use in
  `intel_vllm_triton` but is **not applied** by this feature.
- Every smoke/A/B script `trap`s exit and kills its container; a failed arm
  cannot leave a wedged server holding the GPU.
- Rule: **never leave the node with a broken graph-mode server running.** On
  any fail: stop container → re-run arm A (eager) smoke to prove the GPU and
  weights still serve correctly → record → stop. Eager is the only state the
  node may be left serving in.

## Auto-rollback

Rollback is built into the runner (`deploy/xpu-ornith-graphs-canary/`):

1. **In-script:** startup timeout (no Ready in 900 s), server death, SYCL
   Graph scratch error in the log, or any correctness-smoke failure ⇒ script
   kills the canary container, greps + archives the log, exits non-zero. The
   ladder stops at the first failing arm.
2. **Graph-arm failure ⇒ eager re-verify:** the runner then re-launches arm A
   (eager) and re-runs the short smoke to prove the node is healthy in the
   known-good config before finishing. That result is recorded with the
   failure.
3. **Mid-canary manual trigger:** `docker rm -f ornith-canary` on hal is
   always sufficient — the canary holds no k8s state. Production DS remains
   paused/eager regardless.
4. **Config-level rollback knobs** (same as 01/03): arm C → arm B is
   `VLLM_XPU_GRAPH_FORCE_PIECEWISE=1`; any graph arm → eager is
   `VLLM_XPU_ENABLE_XPU_GRAPH=0` + `--enforce-eager`.

If this canary were ever run inside the DaemonSet instead (05 rehearsal):
pause pattern is the existing label flip (`hal.local/vllm-xpu-paused=true`),
documented in `RESUME-ORNITH-MXFP4.md`; drain/swap before any risky toggle.

---

## Correctness smokes (Ornith-specific; per graph arm, vs arm-A reference)

Ready alone is **not** a pass. Greedy (`temperature=0`) throughout.

| # | Smoke | Hard-fail on |
|---|-------|--------------|
| S1 | Startup + log hygiene | no Ready; `work_group_scratch_memory` / SYCL Graph feature error; cudagraph_mode resolved to NONE; (arm C) missing `FlashAttention-in-graph enabled` line |
| S2 | Short decode (`2+2`, 32 tok) | empty; wrong; garbage |
| S3 | Long decode 512 tok | `!{4,}` loops, NaN-ish garbage, empty, mid-stream collapse |
| S4 | temp=0 repeat ×2 | any byte difference between runs (graph replay nondeterminism) |
| S5 | MoE-routing stress: multi-domain prompt batch (code + math + prose + multilingual), 256 tok each | garbage on any domain; cross-arm gross divergence |
| S6 | Mamba/GDN state stress: ~2k-token context with facts planted early + recall question at the end | wrong/missing recall vs arm A answer (state corruption across long sequence) |
| S7 | Eager-compare: S2/S3/S5/S6 outputs vs arm A | structural divergence (different answer/content). Tail-token drift of a few chars on long generations is tolerated (bf16 near-ties, seen in 03) but must be flagged in results |
| S8 | Post-fail eager re-verify (only after a graph-arm fail) | eager itself failing ⇒ node/weights problem, stop everything |

The known failure signatures are hard fails everywhere: the
`sycl_ext_oneapi_work_group_scratch_memory` RuntimeError, and PIECEWISE-style
silent decode corruption (S3–S7 exist to catch it).

---

## A/B (gate + metrics for the PR)

Same image, same profile, sequential arms on the same GPU; identical greedy
prompt set; single stream ×8 completions of 128 tok after warmup:

- TTFT mean/p50, decode tok/s mean/p50 per arm (A, B, C).
- Output-text agreement A vs B and A vs C on the smoke prompts.
- Startup→Ready wall time and #captured graphs per arm (graph memory/time tax).

Results land in `deploy/xpu-ornith-graphs-canary/results/` (JSON + md) and in
this doc.

## Success / fail criteria — gate for 05

**Canary PASS (mode X promotable):** S1–S7 green on mode X; A/B decode
uplift vs eager recorded; no OOM at the canary profile; nothing in the serve
log suggesting capture fallback that silently disabled graphs.

**Go for 05 (recommend):** arm C pass ⇒ propose FA-in-graph
(`FULL_AND_PIECEWISE`) for cutover; else arm B pass ⇒ propose PIECEWISE;
either way 05 must additionally validate full 128k context + prefix caching +
tool parsers on the DS profile before flipping production. The `is_padding`
in-tree fix ships with whichever mode is promoted (and retires the ConfigMap
runtime patch).

**No-go:** both graph arms fail any smoke ⇒ production stays eager
(`--enforce-eager`, `VLLM_XPU_ENABLE_XPU_GRAPH=0`); record failure modes as
input for kernels/runtime work. The `is_padding` fix is still mergeable (it
is required for eager Ornith on this tree too).

---

## Files on this branch

| File | Change |
|------|--------|
| `vllm/_custom_ops.py` | Remove stale XPU `is_padding`-omitting branches for `topk_softmax` / `topk_softplus_sqrt` (fork kernels require the arg) |
| `deploy/xpu-ornith-graphs-canary/README.md` | Runbook (profile, isolation, rollback) |
| `deploy/xpu-ornith-graphs-canary/env-canary.env.example` | Arm env matrix |
| `deploy/xpu-ornith-graphs-canary/smoke_ornith.sh` | One-arm serve + S1–S7 smokes, auto-kill + eager re-verify hooks |
| `deploy/xpu-ornith-graphs-canary/ab_ornith.sh` | A→B→C ladder + metrics + output-agreement report |
| `deploy/xpu-ornith-graphs-canary/daemonset-canary.yaml.example` | Draft for `intel_vllm_triton` later; **not applied** |
| `docs/xpu/ORNITH_XPU_GRAPHS_CANARY_PLAN.md` | this plan + results |

Kernels: `@aa156578` unchanged (pinned in the 02 image). No kernels branch.

## Ordered tasks + effort

| Task | Est. |
|------|------|
| **T0** Plan (this doc) | done |
| **T1** `is_padding` fix in `_custom_ops.py` | 0.25 d |
| **T2** Canary assets (README/env/smoke/ab/DS-draft) | 0.5 d |
| **T3** hal arm A (eager) — proves fix + Ornith on the 02 image | 0.5 d |
| **T4** hal arm B (PIECEWISE) S1–S7 | 0.5 d |
| **T5** hal arm C (FA-in-graph FULL) S1–S7 | 0.5 d |
| **T6** A/B metrics + results write-up | 0.5 d |
| **T7** Lint, PR with metrics, merge to fork main | 0.25 d |
| **Total** | **~3 d** |

## Risks

| Risk | Mitigation |
|------|------------|
| MXFP4 MoE kernels not capture-safe (silent corruption) | S3–S7 eager-compare gates; PIECEWISE arm isolates MoE-in-graph from FA-in-graph; stop-on-fail ladder |
| GDN/Mamba state desync under graph replay | S6 long-context recall smoke; UNIFORM_BATCH support verified in code but never trusted without smoke |
| Graph memory tax OOMs a 35B on 32 GB Arc | 32k canary context (not 128k), util 0.85; if OOM: reduce `-cc.max_cudagraph_capture_size`, then declare arm fail rather than shrink model guarantees |
| Canary becomes live default by accident | docker-only, port 8021, never touches DS/LiteLLM; DS manifest is a non-applied example; production stays paused |
| Broken Arc left behind | trap-kill in every script + mandatory post-fail eager re-verify (S8) |
| fp8 KV + graphs interaction unproven | fp8 KV kept identical to prod in all arms incl. eager baseline, so any fp8-specific break shows up in arm A first |
| Ready-but-corrupt passes unnoticed | Ready is never the gate; S2–S7 are |

Out of scope: production cutover / DS changes / LiteLLM routes (05),
gpt-oss or any other model, kernels changes, 128k-context validation.

---

## Results (hal, Arc Pro B70)

Image `hal/vllm-xpu:oneapi-2026.0-torch2.13-3deb3160c` + this branch overlaid
(`vllm/_custom_ops.py`, `vllm/envs.py`, `vllm/platforms/xpu.py`,
`vllm/utils/torch_utils.py`) into both the installed and `/workspace` vLLM
copies. `torch 2.13.0+xpu`, `torch.version.xpu=20260000`, kernels
`@aa156578` unchanged. Model `/models/Ornith-1.0-35B-MXFP4`
(compressed-tensors MXFP4 MoE + hybrid GDN/Mamba), bf16, fp8 KV, 32k context,
`--max-num-batched-tokens 4096`, `--max-num-seqs 2`, util 0.85, greedy,
single stream, `ZE_AFFINITY_MASK=0`. Standalone `docker run` — production
DaemonSet never touched.

### `is_padding` fix

Arm A first **failed to boot** on the stock tree with
`_moe_C::topk_softmax() is missing value for argument 'is_padding'` — exactly
the error the production ConfigMap patch works around. With the in-tree fix
(`vllm/_custom_ops.py`) all arms boot cleanly and the runtime patch is no
longer needed. This alone is required for Ornith on this tree, graphs or not.

### A/B (8×128-tok greedy completions after warmup)

| Arm | Ready s | TTFT ms mean | TTFT ms p50 | Decode tok/s mean | Decode tok/s p50 |
|-----|---------|--------------|-------------|-------------------|------------------|
| A: eager (prod config) | 135 | 154.7 | 154.3 | 13.9 | 13.9 |
| B: PIECEWISE | 201 | 72.6 | 71.9 | 72.2 | 72.2 |
| C: FA-in-graph FULL | 180 | 66.6 | 66.4 | 74.0 | 74.0 |

**B vs A:** decode **+419%**, TTFT −53%. **C vs A:** decode **+432%**,
TTFT −57%.

### Correctness (S1–S7, all arms)

- **S1** all Ready; **no `work_group_scratch_memory` / SYCL Graph error** in
  any arm. Arm B captured PIECEWISE (3 sizes, 0.12 GiB, GDN/Mamba ops in
  `splitting_ops` so they stay outside capture). Arm C logged
  `FlashAttention-in-graph enabled` and — because `GDNAttentionBackend` is
  `UNIFORM_BATCH` — auto-downgraded `FULL → FULL_AND_PIECEWISE` (2 decode FULL
  graphs, 0.01 GiB), as designed. Graph memory tax negligible; no OOM at 32k.
- **S2** `2+2 → "4"` on every arm.
- **S3** 512-tok structured explanation coherent on every arm; no `!` loops /
  NaN / collapse.
- **S4** temp=0 repeats **byte-identical within every arm** (short and long).
- **S5** MoE routing (code / math / prose / multilingual) coherent and correct
  on every arm — MXFP4 MoE captured in-graph produces no garbage.
- **S6** GDN/Mamba state recall after ~2k-token filler: all arms recalled
  `MAGNETIC-YELLOW-42` / `Dr. Elena Vasquez` **byte-identically** — no linear-
  attention state corruption under graph replay.
- **S7** eager-compare: short + state-recall outputs byte-identical A/B/C. Long
  free-form generations (S3/S5) share identical prefixes and diverge only at a
  single greedy word-choice branch point ("process" vs "calculation",
  "dew-kissed" vs "frost-kissed"), after which the continuations are fully
  coherent. This is the bf16 near-tie flip 03 documented, **not corruption**
  (similarity metric is low only because one early flip cascades). No arm
  produced garbage on any prompt.

Artifacts: `deploy/xpu-ornith-graphs-canary/results/AB_COMPARE_20260727T153842Z.{json,md}`
plus per-arm `outputs_*` / `arm_*`.

### Go / No-go for 05

**GO.** Both graph arms pass all smokes with large, near-equal decode uplift.
Arm C (FA-in-graph → `FULL_AND_PIECEWISE`) is the recommended 05 mode: it is
marginally faster than B, exercises the full 03 capability, and the GDN
auto-downgrade means it is really "FA + MoE in FULL graphs, GDN/Mamba
piecewise" — the safest full-graph shape available for this model. Arm B
(PIECEWISE) is the fallback if 05's DS-profile validation surfaces any issue.

**05 must still validate on the production DS profile before flipping**: full
128k context (not 32k), `--enable-prefix-caching`, tool/reasoning parsers,
`--max-num-seqs`/util at prod values, and the `is_padding` fix baked into the
image (retiring `patch_topk_is_padding.py`). Production stays eager until then.
