# 05 — Production DaemonSet cutover: Ornith off eager onto XPU graphs

**Branch:** `production-xpu-graphs-cutover` (`krisclarkdev/vllm`; ops changes
live in the `intel_vllm_triton` repo on its same-named branch, merged to its
`main`)
**Base:** fork `main` @ `2d456abf3` (01–04 all merged; 04 gate **GO**)
**Image (pinned):** `hal/vllm-xpu:oneapi-2026.0-torch2.13-2d456abf3` — the 02
image + the four files changed between `3deb3160c` and `main` @ `2d456abf3`
baked into both vLLM copies (`vllm/_custom_ops.py`, `vllm/envs.py`,
`vllm/platforms/xpu.py`, `vllm/utils/torch_utils.py`); exactly the 04 canary
overlay set. Docker `fab6eaea20cd`, k3s digest `sha256:29a69cee7126…`.
**Mode:** FA-in-graph — `VLLM_XPU_ENABLE_XPU_GRAPH=1`,
`VLLM_XPU_GRAPH_FORCE_PIECEWISE=0`, `--attention-backend FLASH_ATTN`,
`-cc.cudagraph_mode=FULL` → auto-resolves to `FULL_AND_PIECEWISE` (GDN/Mamba
is `UNIFORM_BATCH`). 04 canary arm C.
**Status:** **CUTOVER COMPLETE 2026-07-27.** Production `vllm-xpu` DaemonSet
on hal (Arc Pro B70) serves Ornith-1.0-35B-MXFP4 under XPU graphs at the full
production profile. Rollback to eager proven live in 156 s.

---

## What changed in production

This feature contains no vLLM code change — it is the deliberate production
enablement of 01–04 on the `hal` k3s DaemonSet (manifests + runbooks in
`intel_vllm_triton`):

- Image `kris-fork-577e1a932` (pruned) → pinned `…-2d456abf3`; imported into
  k3s via `docker save | k3s ctr images import -`, pod imageID verified.
- `--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0` → graphs flags above.
- `patch_topk_is_padding.py` runtime ConfigMap patch **retired**: the in-tree
  `is_padding` fix (04, PR #5) boots Ornith clean — verified in production
  (phase E booted eager on the new image without the patch).
- Everything else unchanged: port 8004, `--max-model-len 131072`,
  `--max-num-batched-tokens 4096` (Mamba align), `--max-num-seqs 2`, util
  0.9, bf16, fp8 KV, prefix caching, qwen3_xml tool parser, qwen3 reasoning
  parser.
- Rollback manifest (`vllm-xpu-daemonset.eager-rollback.yaml`) uses the same
  new image with eager flags — the old eager image no longer exists on hal.

## Validation at the production profile (04's open items)

04 validated 32k context without parsers; 05 additionally proved under
graphs: full 131072 context serving (no OOM; graph memory tax negligible —
3 PIECEWISE + 2 FULL decode graphs), a ~70k-token in-context recall probe,
`--enable-prefix-caching` (repeat long prompt: 45 s → 1.6 s TTFT),
auto tool choice returning well-formed `tool_calls`, and the reasoning
parser (`reasoning_content` present via the LiteLLM thinking route).

## A/B metrics (production DS profile, 8×128-tok greedy single-stream after warmup)

| Phase | TTFT ms mean/p50 | Decode tok/s mean/p50 | Long-ctx ~70k first/repeat |
|---|---|---|---|
| E: eager (rollback config, new image) | 148.9 / 149.9 | 14.3 / 14.3 | 47.5 s / 2.0 s |
| G: graphs (FA-in-graph, `FULL_AND_PIECEWISE`) | 68.5 / 67.6 | 73.9 / 73.9 | 45.0 s / 1.6 s |

**Decode +416.8%, TTFT −54.0%** — consistent with the 04 canary
(+432% / −57% at the smaller canary profile).

Correctness gates (both phases, plus cross-phase compare): S2 short, S3 long
(no `!` loops / NaN / collapse), S4 temp=0 byte-determinism, S5 MoE
multi-domain, S6 ~2k-token GDN state recall, ~70k-token recall, tool-call
parse — all green. Cross-phase (S7): byte-identical on deterministic prompts
(S2, S5_code, S6, 70k recall); long free-form generations diverge only at
single greedy bf16 near-tie branch points and continue coherently (03/04
documented behavior, not corruption). No `work_group_scratch_memory` / SYCL
Graph error in any serve log.

## Rollback (proven live)

```bash
sudo k3s kubectl -n inference apply -f vllm-xpu-daemonset.eager-rollback.yaml
sudo k3s kubectl -n inference delete pod -l app=vllm-xpu   # OnDelete
```

Drill: graphs → eager Ready + correct decode in **156 s** (`enforce_eager:
True` verified in the new pod log); eager → graphs restore in 298 s. Both
inside the 20-min time box. Triggers: the SYCL scratch RuntimeError
signature, decode corruption / NaN / `!{4,}` loops, temp=0 nondeterminism,
OOM/crashloop. Operational gotcha discovered: with hostNetwork, a
*terminating* pod still answers `:8004/health` — verification must confirm
the new pod name before trusting health.

## Artifacts

`intel_vllm_triton` repo (branch `production-xpu-graphs-cutover`, merged):
`deploy/ORNITH-XPU-GRAPHS-CUTOVER-PLAN.md` (plan + full results),
`deploy/vllm-xpu-daemonset.yaml` (graphs SoT),
`deploy/vllm-xpu-daemonset.eager-rollback.yaml`,
`deploy/xpu-graphs-cutover/validate_prod.sh`,
`deploy/xpu-graphs-cutover/results/{phase_E_CUTOVER,phase_G_CUTOVER,AB_CUTOVER}.json`,
updated `deploy/RESUME-ORNITH-MXFP4.md`.
