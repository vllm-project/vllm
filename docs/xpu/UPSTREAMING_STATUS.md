# Upstreaming status — XPU graphs feature (as of 2026-07-27)

Duplicate-work checks run per AGENTS.md (`gh issue view` / `gh pr list` on
`vllm-project/vllm` and `vllm-project/vllm-xpu-kernels`). Outcome: of the
planned PR series (A–D + kernels track), **only PR-A is new work**. Everything
else is already open or already merged upstream.

## Already upstream — do not duplicate

| Planned | Status |
|---------|--------|
| PR-B: oneAPI 2026.0 + torch 2.13 image bump | **Duplicated** by vllm-project/vllm#48677 ([XPU] upgrade to torch 2.13, yma11, draft; touches `docker/Dockerfile.xpu`, `requirements/xpu.txt`, `requirements/test/xpu.txt`, `setup.py`) and #47709 (oneAPI env sourcing in Dockerfile). Our unique value = the build fixes in `ONEAPI_2026_BUILD_NOTES.md` (2026.0-vs-2026.1 icpx pin, SYCL_HOME for kernels cmake, oneCCL so.9) and A/B metrics — offer as review input on #48677, not a competing PR. |
| PR-C: `is_padding` removal in `_custom_ops.py` | **Handled by Intel's WA cycle.** Official kernels merged `is_padding` (vllm-project/vllm-xpu-kernels#481, 2026-07-24, after the v0.1.11.1 release that vLLM pins). Upstream is actively adding/removing these WAs (#49395, #49884) and will drop them when the next kernels wheel is pinned. Do not open; revisit only if the WA removal has not landed after the next kernels release. |
| PR-C: softcap/ALIBI forwarding + PR-D: MXFP8 oracle prefer-XPU | **Already open from this fork:** vllm-project/vllm#49813 ([XPU] Prefer MXFP8 MoE XPU backend; forward softcap/ALIBI). intel-ci **passed**, DCO passed, reviewers cc'd by jikunshang. The only failing check (`pre-run-check`) is a permissions gate: first-time author needs a maintainer to add the `ready` label — human action: ask in the PR / #pr-reviews Slack. |
| `VLLM_XPU_ATTN_ALLOW_FALLBACK` env | **Not upstreamable now** — read by the fork kernels only (no counterpart in official kernels). Follows the fail-closed kernels PRs below. |
| Kernels track (fail-closed attn, softcap/ALIBI, Split-K, MXFP8 MoE) | **Already open on vllm-project/vllm-xpu-kernels:** #485 (Split-K mix-batch decode), #487 (fail closed by default on missing FA2 shapes), #488 (native Xe2 MXFP8/block-FP8 MoE), #489 (fail closed on softcap/ALIBI). |

## PR-A — ready to open (the one new contribution)

Branch: `upstream-pr/xpu-fa-in-graph` (pushed to `krisclarkdev/vllm`, single
commit on top of `upstream/main`, DCO signed). Diff:
`vllm/utils/torch_utils.py` (`supports_xpu_fa_in_graph()`),
`vllm/platforms/xpu.py` (clamp / fail-closed / enable logic),
`vllm/envs.py` (`VLLM_XPU_GRAPH_FORCE_PIECEWISE`),
`tests/utils_/test_torch_utils.py` (6 param cases — pass in the
oneAPI 2026 container, no GPU needed).

Related upstream context to link: intel/torch-xpu-ops#3142 (the scratch-in-
graph error), vllm-project/vllm#48946 (MXFP4 MoE corrupt under piecewise XPU
graphs on PVC — our Arc validation is a useful datapoint), #48677 (torch 2.13
bump this feature benefits from).

**Human steps (agent PRs are not allowed):** review every line of the branch,
open the PR with the description below, attach the 3-way sweep numbers once
the bench completes (placeholders marked), state AI assistance.

### Draft PR description

```markdown
## Purpose

FlashAttention SYCL kernels use `sycl_ext_oneapi_work_group_scratch_memory`,
which SYCL Graph cannot capture before oneAPI 2026.0
(intel/torch-xpu-ops#3142). On today's runtimes, enabling XPU graphs with
FLASH_ATTN and a full cudagraph mode crashes at warmup with
"work_group_scratch_memory ... not yet available for use with the SYCL Graph
extension"; the current guidance is "FLASH_ATTN supports PIECEWISE only".

This PR makes full-graph FlashAttention capture available where the runtime
supports it, and safe everywhere else:

- `supports_xpu_fa_in_graph()`: `torch.version.xpu >= 20260000`.
- `VLLM_XPU_GRAPH_FORCE_PIECEWISE` (default **on**): clamps full graph modes
  to PIECEWISE, preserving today's behavior.
- Opt-out on a capable runtime keeps the requested FULL /
  FULL_AND_PIECEWISE mode and logs "FlashAttention-in-graph enabled".
- Opt-out on an incapable runtime falls back to PIECEWISE with a warning
  instead of crashing at warmup (fail closed).

## Test Plan

- `pytest tests/utils_/test_torch_utils.py -k supports_xpu_fa_in_graph`
  (6 cases, no GPU required).
- Serve-level validation on Intel Arc Pro B70, torch 2.13.0+xpu
  (`torch.version.xpu=20260000`), oneAPI 2026.0 image:
  - dense (Qwen2.5) and MXFP4-MoE + hybrid-GDN 35B, arms: eager /
    PIECEWISE / FULL (auto-resolves to FULL_AND_PIECEWISE for hybrid
    UNIFORM_BATCH backends);
  - greedy determinism, long-decode, MoE-routing and linear-attention
    state-recall smokes, all byte-compared vs eager.

## Test Result

- Unit tests pass.
- No SYCL Graph scratch error in any arm; on oneAPI 2025.3 runtimes the
  fail-closed path clamps to PIECEWISE as intended.
- 35B hybrid MoE, single stream, 8x128-token greedy: decode 13.9 -> 74.0
  tok/s (+432%), TTFT 154.7 -> 66.6 ms (-57%) for FA-in-graph vs eager;
  byte-identical short/state-recall outputs, identical-prefix long outputs
  (one bf16 near-tie word flip, coherent continuations).
- <PLACEHOLDER: 3-way sweep table (concurrency x length grid: TTFT/TPOT/ITL
  percentiles, prefill+decode throughput, derived TFLOPS) vs graphs-off and
  vs the pre-change baseline image — deploy/xpu-graphs-bench in the fork.>

AI assistance was used to develop and validate this change; every line has
been reviewed by the submitter.
```

## Fork-only material (never upstreamed)

`deploy/xpu-*` canary/bench dirs, `docs/xpu/*` plan docs and results,
hal-specific image tags, fork-kernels git pin in `requirements/xpu.txt`.
