# Genesis `_genesis/` Package Changelog

> **Cross-reference:** this is the **engineering log** (per-commit, per-A/B,
> per-decision detail). Public-facing release notes live in the top-level
> [`../../CHANGELOG.md`](../../CHANGELOG.md). Both should agree on version
> labels — the engineering log is conventionally one version ahead when
> there's in-flight work between releases.

## v7.72 — 2026-05-05 (audit hardening + 7 new patches PN59-PN67 sprint)

This is the engineering label for what the public CHANGELOG calls the
"v7.72 sprint" — see public [`../../CHANGELOG.md`](../../CHANGELOG.md)
§"v7.72" for the operator-facing summary. This entry only adds the
engineering-level delta beyond what the public log captures.

### Engineering-only delta

- **Schema validator** extended to recognize `retire_after_pin` field
  (added 2026-05-05 alongside vllm#39931 P4 supersede marker).
- **Status-helper migration via AST**: bulk-injection script
  (`/tmp/inject_skipped_branch.py`, regex-driven) inserted `if result
  == TextPatchResult.SKIPPED: return "skipped", ...` branch in 28
  wiring files in a single batched pass. 3 files were skipped by the
  script (`patch_63_mtp_gdn_state_recovery.py`, `patch_38b_compile_safe_hook.py`,
  `patch_28_gdn_core_attn.py`) because they already had a SKIPPED
  branch via different code path.
- **MultiFilePatchTransaction `_dry_run`** now does sequential preview
  (mutates `preview = src` through each sub-patch's replacement BEFORE
  checking the next anchor) so an early sub-patch that invalidates a
  later anchor is caught at dry-run time. Plus anchor-uniqueness check
  (`count(anchor) > 1` → ambiguous_anchor SKIPPED). Closes audit P1.2.
- **PN40 per-method auto-disable** refactored from `set[str]` to
  `set[tuple[str, str]]` keying. New primitive
  `auto_disable_sub_kernel_for_method(method, sub, reason)` — old
  `auto_disable_sub_kernel(sub, reason)` shim iterates over
  `("mtp", "dflash", "ngram", "eagle")` so existing callers keep working.
  `is_auto_disabled(sub, method=None)` preserves legacy "any method" semantics
  when method is None. Closes audit P1.6 / A-11.
- **`pytest_collection_modifyitems`** in `tests/conftest.py` now AST-scans
  every test file at collection time for module-level `import torch` and
  auto-applies `requires_torch` skip — closes A-15.
- **A-19 audit test** (`test_a19_optional_sub_patches_marker_policy.py`)
  ships with 4-patch allowlist documenting graceful-degradation cases
  (P24/P27/P59/P83) where partial-apply is INTENTIONAL (not the PN40
  all-or-nothing trap).
- **GdnScratchPool docstring honesty pass**: opening paragraphs now
  explicitly admit the production driver doesn't call `acquire_*` —
  the WINDOWING delivers the −142 MiB/GPU saving, not the POOL reuse.
  POOL primitives remain as future-infra + reference utility test
  surface. Closes audit P2.5.
- **PN59 streaming-GDN tightened eligibility** in
  `kernels/streaming_gdn_driver.py`: now also rejects calls with
  non-trivial `chunk_indices` / `chunk_offsets` metadata (was silently
  dropping). Plus `GENESIS_PN59_DEBUG=1` env-gated bypass-reason logging
  for debug. Closes audit P2.4.
- **PN65 v2 uvicorn dedup via Filter**: `_DropUvicornAccessInfo`
  `logging.Filter` attached to BOTH root logger AND `uvicorn.access`
  logger (belt-and-suspenders for uvicorn re-init after our middleware
  install). v1's `setLevel(WARNING)` was bypassed by uvicorn's late
  `log_config` re-instantiation. Closes audit P2.1.
- **PN61 v2 pre-emptive language_model_only**: `_wrap_load_weights` now
  detects compressed-tensors / NVFP4 quant_method on qwen3_vl + sets
  `language_model_only=True` BEFORE `original_load_weights` runs, so
  the loader doesn't take the ViT branch and produce partial state.
  Post-failure handler stays as final safety net. Closes audit P2.3.
- **P37 / P40 / P5b / P7b env_flag aligned**: dispatcher registry now
  uses short forms (`GENESIS_ENABLE_P37`, etc.) matching wiring code
  + launch scripts. `env_flag_guard` no longer reports our PROD env as
  suspicious typo.
- **`env_flag_guard._ALLOWLIST_PREFIXES`** extended with `"GENESIS_DISABLE_"`
  so disable-prefix typos are now scanned (was unreachable). Closes audit P2.10.
- **`gdn_composability.find_composability_warnings`** now honors
  `composes_with` (skip warning if pair explicitly declared compatible)
  and conflict-message picks the side that DECLARED the conflict (was
  always pa). Closes audit P3.1.
- **`compat/preflight_checks.check_spec_decode_token_loop`** rewritten
  as state-machine (`current_streak` / `max_streak`) — was summing all
  matches in window which gave false positives on interleaved snapshots.
  Closes audit P2.9 / club#34.
- **PN64 env-gate enforcement**: `marlin_tuning.py:get_optimal_block_size_m/
  num_warps/num_stages` all check `_pn64_enabled()` before returning the
  `(12, 0)` table entry. Audit P1.7 closed.
- **PN40 scheduler subpatch split**: separate `GENESIS_PN40_SCHED_OBSERVE_MARKER`
  + `GENESIS_PN40_SCHED_K_TRIM_MARKER` markers + separate TextPatcher
  instances. Backwards-compat `_make_scheduler_patcher()` shim returns
  observe-only patcher. Audit P1.5 closed.
- **`gdn_scratch_pool.GdnScratchPool.should_apply()`** now accepts unified
  bool set (`"1","true","yes","y","on"` case-insensitive) matching all
  other Genesis env flags. Audit P3.3 closed.
- **PN66 + PN67 backports**: vllm#41696 (multiturn `</think>` leak in
  DelegatingParser, panpan0000) + vllm#41674 (thinking_budget inverted
  bool single-token fix, JasonKeyiL) — both wired with TDD; opt-in.

## v7.71 — 2026-05-04 EOD (PN40 omnibus DFlash optimization — sub-A shipped)

After PN37 attempt (DFlash tiny-Q non-causal Triton kernel) was empirically
disproved as a viable replacement for FA2 attention forward (microbench
showed 0.33-0.73x slower vs torch SDPA which routes to FA2 internally),
deep research identified 3 alternative angles that don't compete with
already-optimal FA2: drafter weight quant (PN38 candidate, PR #40425),
adaptive DFlash N (PN39 candidate, SGLang tier policy), and fused
per-layer K/V operations (PN40).

### NEW: PN40 — DFlash drafter omnibus (sub-kernel A shipped, B/C/D in design)

Strict-superset DFlash drafter optimization. v1 ships **sub-kernel A only**:
fused per-layer K-norm Triton kernel that replaces the L-iteration
`for i in range(L): ops.rms_norm(...)` loop in `qwen3_dflash.py:397-404`
with a single kernel launch.

**Numerical TDD**: 12/12 PASS, **rel_avg = 0.0000** (bit-equivalent vs
sequential reference). Tested shapes:
- 27B drafter: L=5, H={2,4}, N={16, 256, 1024, 4096}
- 35B drafter: L=8, H={1,2}, N={16, 256, 1024, 4096}

**Honest microbench vs `vllm._custom_ops.rms_norm`** (real CUDA kernel,
50-200 iter after warmup, A5000 SM 8.6):

| Config | N | PN40 µs | REF µs | Δ saved | Speedup |
|---|---|---|---|---|---|
| 27B L=5 | 16 | 17.10 | 55.14 | +38.04 | **3.22x** |
| 27B L=5 | 256 | 17.48 | 53.76 | +36.28 | **3.08x** |
| 27B L=5 | 1024 | 17.07 | 54.52 | +37.46 | **3.19x** |
| 35B L=8 | 16 | 17.53 | 87.26 | +69.73 | **4.98x** |
| 35B L=8 | 256 | 16.84 | 87.06 | +70.22 | **5.17x** |
| 35B L=8 | 1024 | 16.38 | 87.12 | +70.74 | **5.32x** |

**E2E verification on live serving** (boot + tool_call regression):
- **27B+DFlash N=5 + PN40 ON**: 0 ERR, TPS 67-118 (vs baseline 71-112,
  **+5.5% peak**), tool_call works
- **35B+DFlash N=3 + PN40 ON**: 0 ERR, TPS 102-141 (vs baseline 102-140,
  same), **4/4 tool_call clean** (auto + required)

**Strict no-regression contract** (Sander explicit requirement):
- Eligibility predicate cheap (no GPU sync): L ∈ [2,16] + D ∈ {64,128} +
  BF16/FP16 only
- try/except wraps integration site → ANY failure falls through to
  baseline per-layer loop (preserved verbatim)
- Default OFF (env-gated `GENESIS_ENABLE_PN40_DFLASH_OMNIBUS=1`)
- Composes additively with PN21/PN23/PN24 (different anchor surfaces)

**Files added**:
- `vllm/_genesis/kernels/pn40_dflash_omnibus.py` (191 lines)
- `vllm/_genesis/wiring/spec_decode/patch_N40_dflash_omnibus.py` (130 lines)

**Lessons learned (PN37 → PN40 pivot)**:
- Don't compete with FA2 on attention forward — torch SDPA correctly
  routes to FA2 packed-GQA path even for tiny Q
- Reduce kernel launch overhead in OTHER hot paths (per-layer loops)
- TDD + microbench BEFORE full integration spared 8-12h of wasted wiring
  on PN37 broken design

**Sub-B/C/D landed same day** (35/35 logic TDD PASS):

- **Sub-B `PersistentKVBufferPool`**: LRU-bounded per-shape buffer cache
  with `max_entries_per_shape=4` + `max_distinct_shapes=16`; hit-rate
  tracked. DFlash-specific MVP (MTP allocations torch.compile-handled).
- **Sub-C `AdaptiveSpecKController`** (UNIVERSAL — applies to MTP K + DFlash N):
  EMA acceptance-length tracking (α=0.2), tier-policy hysteresis
  (up≥0.85·K, down≤0.55·K). Default tiers: `mtp_3=[0,1,3]`,
  `dflash_5=[0,1,3,5]`, `dflash_3=[0,1,3]`. NaN-trip safety. 10-step
  warmup grace prevents cold-start oscillation.
- **Sub-D `StabilitySentinel` + `classify_workload`** (UNIVERSAL — all 4 configs):
  Sliding-window AL drop detector (window=50, threshold=0.5 vs slow EMA
  α=0.05). Workload classifier: `code` (tool_call/fim/def/fence sigs),
  `long_ctx` (≥16K tokens), `short_ctx` (<1K), `free_form` (default).

Per-sub env toggles `GENESIS_PN40_ENABLE_SUB_{A,B,C,D}=0` to disable
individually. Master env `GENESIS_ENABLE_PN40_DFLASH_OMNIBUS=1`.

**MTP applicability** (для 27B PROD + 35B PROD MoE):

| Sub | DFlash | MTP PROD | Universal? |
|---|---|---|---|
| **A** fused per-layer K-norm | ✅ shipped | ❌ N/A (`num_mtp_layers=1`) | DFlash-only |
| **B** persistent buffer pool | ✅ MVP | ⚠️ torch.compile-handled | DFlash-only |
| **C** adaptive K controller | ✅ N=3,5 | ✅ **K=3 [0,1,3]** | UNIVERSAL ⭐ |
| **D** workload classifier + sentinel | ✅ | ✅ | UNIVERSAL ⭐ |

**Wiring status**:

- Sub-A: text-patch into `qwen3_dflash.py` SHIPPED, validated 27B+35B DFlash
- Sub-B/C/D: orchestrator API ready (`get_buffer_pool()`,
  `AdaptiveSpecKController(tiers, base_k)`, `StabilitySentinel()`,
  `classify_workload()`, `orchestrator_status()`). Production-wiring
  for MTP/DFlash spec-decode call sites deferred to follow-up sprint
  (call sites: `eagle.py` proposer step + accepted-length feedback hook
  from scheduler `update_from_output`).

### Removed: PN37 (kept as research artifact)

PN37 wiring removed; standalone kernel kept at
`vllm/_genesis/kernels/pn37_dflash_tiny_q_attn.py` for future SM 8.9+
retest. Lifecycle marked `research_artifact` in PATCH_REGISTRY. Premise
disproven via microbench (0.33-0.73x slower than FA2 in most regimes).

## v7.70 — 2026-05-04 (vllm pin bump + path-fix sweep)

vllm runtime bumped from `0.20.1rc1.dev16+g7a1eb8ac2` (2026-04-23) to
`0.20.2rc1.dev9+g01d4d1ad3` (2026-05-04 nightly). Test container booted
clean with full PROD env-set; smoke-test green (Boot + `/v1/models` +
chat completion + tool_choice=required `get_weather({"city":"Paris"})`).

### vllm pin reference (last 3 commits on the bumped HEAD)

| SHA (short) | UTC | Author | Title | Files |
|-------------|-----|--------|-------|-------|
| `6ec9bbe` | 2026-05-04 05:22 | Andreas Karatzas | [CI] Stabilize cpu offload compressed tensors test (#41102) | `tests/quantization/test_cpu_offload.py`, `tests/utils.py` |
| `6f53753` | 2026-05-04 10:37 | Stefano Castagnetta | [Bugfix] Apply ruff-format to hyperclovax.py (#41620) | `vllm/transformers_utils/configs/hyperclovax.py` |
| `62ba751` | 2026-05-04 11:47 | Stefano Castagnetta | Revert "[Doc] Fix RTD build: pytorch.org/docs/stable/objects.inv 404" (#41618) | `mkdocs.yaml` |

(All three are CI/format/docs — none touch our hot path. Listed here so
future audits can confirm "engine matches the latest version we shipped on".)

### Two new-pin regressions located + closed

1. **`turboquant_attn.py:597 query_start_loc.tolist()` cudagraph crash** —
   upstream regression in this pin only. Closed by enabling existing **P78
   v6** (`GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1`). All 4 sub-patches
   (Site B early-return guard + Sites C/D/E metadata-builder pre-compute)
   anchor cleanly on the new file. PROD's previous `P78=0` was correct
   for the old pin; v7.70 flips it to `=1` in the start scripts.

2. **`arg_utils.py:1706 NotImplementedError` for TQ + hybrid** — anchor
   itself unchanged (P4 still applies cleanly), but the new pin's
   plugin-load order means runtime text-patch happens AFTER the module
   is already imported in the API-server process. Mitigation: explicit
   `python3 -m vllm._genesis.patches.apply_all` PRE-pass in the start
   script, BEFORE `exec vllm serve`. PROD scripts already had this; one
   archived test script was missing it (now fixed).

### Patch-Audit Methodology applied this session

(Per Sander rule 2026-05-04: "не просто отключать патчи а смотреть что
мержанули и как, в чем отличие с нашими патчами".)

| Patch | Upstream PR | Verdict | Rationale |
|-------|-------------|---------|-----------|
| **P8** | (refactor) | RETIRE | upstream refactored `_report_kv_cache_config` to use `get_max_concurrency_for_kv_cache_config`; same problem, different solution. Apply now silently `_skipped("retired ...")`. |
| **PN13** | #41235 (Roi Koren) | RETIRE | byte-identical backport — upstream merged the same `lambda *args, **kwargs: None` change |
| **PN19** | #41268 | KEEP | OPEN, not merged — anchor still valid |
| **P85+P83** | (config quirk) | DISABLE in 27B-PROD only | Both fire only when `--enable-prefix-caching` is set; 27B Lorbus disables prefix-caching to avoid DS conv state crash → patches are dead code. Validator was emitting `P85 requires P84` ERROR every boot. Disabling silences the error without changing runtime behavior. |

### NEW: PN36 — back-compat alias for upstream attribute rename

The new pin renamed `StructuredOutputManager.reasoner` → `reasoner_cls`
in `__init__` but 5 call-sites in the same file still read
`self.reasoner`. Triggered by ANY structured-output request under
spec-decode (e.g. `tool_choice="required"`, `response_format`):

```
File ".../vllm/v1/structured_output/__init__.py", line 429,
  in identify_constrained_draft_tokens
    if self.reasoner is None:
AttributeError: 'StructuredOutputManager' object has no attribute 'reasoner'
```

**PN36** text-patches `__init__` to also bind `self.reasoner = None`.
The 5 affected call-sites all guard with `if self.reasoner is None:`,
so a None alias makes them short-circuit cleanly (degraded
reasoning-aware spec partitioning) instead of crashing the engine.
Default ON — no regression risk because the upstream code is already
broken without it. No upstream PR yet (file new issue when nightly
bumps next).

### NEW: vllm pin allowlist gate (Sander request — "защита от дурака")

`vllm/_genesis/guards.py` now exposes `KNOWN_GOOD_VLLM_PINS` +
`assert_vllm_pin_allowed(policy="warn"|"strict")`. `apply_all.main()`
calls the gate FIRST before any text-patch lands. All 4 PROD start
scripts now set `GENESIS_VLLM_PIN_POLICY=strict` and echo
`pip show vllm | head -3` for visible pin in boot log.

When a new vllm pin is qualified, add the full version string (e.g.
`0.20.2rc1.dev9+g01d4d1ad3`) to `KNOWN_GOOD_VLLM_PINS` and document
the validation surface in this changelog. Strict mode `sys.exit(2)`
on unknown pin — no silent foot-gun.

## v7.68 — 2026-05-02 (cross-rig diagnose + fix)

After v7.66 reached noonghunna's club-3090 single-card and dual-card
3090 rigs, three distinct bug classes surfaced that 2× A5000 PROD
didn't reproduce. All three are root-caused and fixed in v7.68; one
new companion patch (PN34) ports a noonghunna setup-time sidecar
into Genesis as a first-class opt-in.

### PN30 v7.68 — dst-shaped temp (replaces v7.65 layout path)

The v7.65 `get_conv_copy_spec` patch built a compact temp via
`state[src_block_id, :, offset:].contiguous()`. That returns a
`(dim, state_len - offset)` tensor where the row-stride equals
`state_len - offset`, but the destination block in the conv state
pool has row-stride `state_len`. A raw memcpy into the destination
packed source rows into the wrong destination column offsets,
silently corrupting DS conv state on every `offset > 0` continuation
copy.

The user-visible failure was a TQ store CUDA assert several layers
downstream — the assert pointed at the wrong layer, which is why
diagnosis took several rounds with noonghunna's logs and a
ChatGPT/Codex CLI cross-check (club-3090 commit `9af1a52`).

Fix: 3-file text-patch with new part3 patching
`collect_mamba_copy_meta` to build a dst-shaped temp via
`state[dest_block_id].clone()` + tail copy. Old part1 path is now
fail-closed `RuntimeError` so any caller still routing through it
gets a loud failure instead of silent corruption. Marker bumped
v7.65 → v7.68. 12 TDD tests (8 original + 4 new layout-correctness).

### PN25 v7.68 — import-time registration

v7.66's `direct_register_custom_op` switch fixed the v7.65 fork-spawn
crash on TP=2 but still failed on TP=1 spawn:

```text
torch._dynamo.exc.Unsupported:
  instantiate_user_defined_class_object torch.library.Library
```

Root cause: `Library("genesis", "FRAGMENT")` was constructed inside
the worker's first Dynamo trace context. v7.68 text-patches
`activation.py` itself to register the custom op at module-import
time (BEFORE any trace can happen) and caches the result as the
module-global `_GENESIS_PN25_SILU_AND_MUL_OP`. `forward_native` body
then reads only the cached global — never registers, never enters
the Library code path under trace.

Same import-time pattern preventively applied to **P7b**
(`gdn_dual_stream_customop`) — same bug class would fire on any
TP=1 enable. Pattern credit: noonghunna's
`patch_pn25_genesis_register_fix.py` (club-3090 commit `a62ad78`).

### PN34 (NEW) — workspace lock runtime relaxation

PN33 fixed BOOT-time `_dummy_sampler_run` under-counting. The
runtime decode path on `turboquant_attn.py:1350:_decode_attention`
still raised `WorkspaceManager._ensure_workspace_size` AssertionError
on rare paths (continuation-prefill into long context, MTP K=3 +
decode mid-stream) — same lock invariant, different trigger.

PN34 ports noonghunna's `patch_workspace_lock_disable.py`
setup-time sidecar directly into the Genesis registry: relaxes the
strict assertion to one-shot `WARN+grow-anyway`. Default OFF
(it relaxes a strict-debug invariant; operators opt in only when
they hit the runtime case). Requires PN33. Will retire when
vllm#40706 (TQ scratch dedup + reserve-worst-case at warmup)
merges upstream.

### P103 fix — undefined `T` in chunked_fwd loop

Latent since v7.62.20 ship: `wiring/hybrid/patch_103_fla_cliff2_chunked.py:197`
used bare `T` in `for start in range(0, T, _MAX_T)` without ever
defining it. Cliff 2 chunked path silent-crashed `NameError` on
every trigger. PROD didn't surface this because continuous batching
keeps `q.shape[1] ≤ max_num_batched_tokens (4096)`, well under the
`_MAX_T = 32768` threshold that gates the chunked branch — so the
buggy code never executed in our config.

Fix: `T = q.shape[1]` immediately before the loop, plus dropped two
unused locals (`BT`, `is_last`). Found by Gemini static-analysis
audit (see "Audit pass" below).

### Live validation matrix (2× A5000, 2026-05-02)

| Config | Boot | TPS @ 256t | Tool-call | Active patches |
|---|---|---|---|---|
| 27B INT4 + TQ k8v4 + MTP K=3 (PROD) | OK | 104.0 (CV 0.5%) | clean | PN33 + PN25(v7.66) + 45 |
| 35B-A3B FP8 + MTP K=3 | OK | 183.7 | clean | PN33 + PN25 + PN26b + PN8 |
| 35B-A3B DFlash | OK | 155.0 | clean | PN33 + PN22 + PN23 + PN24 + PN8 |
| 27B INT4 + DFlash drafter K=5 | OK | 129.3 | clean | PN33 + PN22 + PN23 + PN24 + PN12 + PN17 |

PN33 text-patch verified live in `gpu_model_runner.py` on all 4
configs (marker present + K-aware code in place). 27B+drafter
result lines up with noonghunna's published 78 narr / 128 code on
2× 3090 — same drafter recipe, similar consumer Ampere → cross-rig
reproducibility confirmed.

---

## v7.67 — 2026-05-02 (REJECTED on live test)

Tried `@torch.compiler.disable` on `SiluAndMul.forward_native`
(SGLang pattern from
`python/sglang/srt/layers/attention/triton_backend.py`). Empirically
failed on 27B + TQ k8v4 + MTP K=3 boot:

```text
torch._dynamo.exc.Unsupported:
  logging.Logger method not supported for non-export cases
```

Stack showed Dynamo tracing INTO `forward_native` body despite the
decorator, hitting `log.info()` inside `acquire_silu_out`. Hypothesis:
`@torch.compiler.disable` on a `@staticmethod` accessed through
vLLM's `custom_op._forward_method` dispatcher does NOT propagate —
the dispatcher reaches the underlying function via `getattr` which
bypasses the decorator's frame-guard. SGLang's working
`@torch.compiler.disable` patterns are on module-level functions,
not `@staticmethod` on classes called via dispatchers — pattern
doesn't transfer.

Reverted to v7.66 in commit `d585d7d`.

---

## v7.66 — 2026-05-02 (root-cause spec-decode warmup fix)

### PN33 — spec-decode warmup K-aware sizing (default ON)

Backport of vllm-project/vllm#37521 by `itailang` (OPEN at backport
time) EXTENDED beyond its `use_eagle()` gate to cover MTP, ngram,
and draft-model methods. The vanilla warmup uses dummy K=1 draft
tokens regardless of real `num_speculative_tokens`, under-counting
the rejection-sampler footprint at profile time.

Same root cause produces TWO distinct symptoms in the wild:

- **(a)** ampersandru's mid-stream OOM via `propose_draft_token_ids`
  (memory pool sized for K=1 explodes when real K=3 fires).
- **(b)** noonghunna's workspace-lock AssertionError on dev205 + MTP
  K=3 single-card (workspace reserved for K=1 then `_ensure_size`
  asserts when K=3 grows it post-lock).

PN33 fixes the root. Default ON, opt-out via
`GENESIS_DISABLE_PN33_SPEC_DECODE_WARMUP_K=1`. 12 TDD tests pass.
Drift markers tightened in commit `fc89395` to PR-#37521-specific
lines after v1 false-positive matched generic vllm code.

### PN25 v7.66 — direct_register_custom_op refactor

Switched `silu_and_mul_pooled` and `dual_linear_parallel`
registration from `@torch.library.custom_op` to vLLM canonical
`direct_register_custom_op` from `vllm/utils/torch_utils.py:899`.
`Library("genesis", "FRAGMENT")` at module level. Same fork-safe
`hasattr()` pre-check guard as v7.65.

Schema introspection happens at module import (synchronous, before
any Dynamo trace), eliminating the "infer_schema skipped frame"
crash class — for TP≥2 spawn. TP=1 still crashed (different code
path); v7.68 PN25 closes that gap.

### PN32 — GDN chunked-prefill (Cliff 2 single-24GB-GPU OOM fix)

Splits GDN `forward_cuda` core attention + post-projection into
chunks of 8K (default) when `num_tokens > 16K` (default; both
env-tunable via `GENESIS_PN32_CHUNK_SIZE` and
`GENESIS_PN32_THRESHOLD`). Closes >50K-token single-shot OOM on
1× 3090 / 4090 / 5090.

Conflicts with P28 (legacy persistent buffer pool) — operator picks
one. Documented in dispatcher entry, wiring docstring, and new
test `test_pn32_documents_p28_conflict`. Default OFF — cross-rig
validation needed (our 2× A5000 PROD doesn't hit Cliff 2 threshold).

---

## Audit pass — 2026-05-02 (Gemini + ChatGPT/Codex CLI)

Two independent static-analysis audits against the genesis-vllm-patches
tree to catch latent issues that pytest + live-boot couldn't surface
(rare exception paths, torch-less import, doc/code drift).

| Audit | Tool | Findings | Real bugs | Commit |
|---|---|---|---|---|
| 1 | Google Gemini | 1 critical | 1 (P103) | `5743c03` |
| 2 | ChatGPT/Codex CLI | 16 (G-001..G-016) | 9 | `82c64c1` + `6f9c5eb` |

Real-bug summary (full per-finding writeup in top-level
[`CHANGELOG.md`](../../CHANGELOG.md)):

- G-001 — `model_detect.py:185` undefined `base` in exception path,
  masked by dispatcher's "conservative apply" fallback. NameError
  could have flipped a genuine model-incompat into "apply patches
  anyway" → hybrid GDN patches on a non-hybrid model. Fixed:
  `base` → `source_label`.
- P103 (Gemini) — undefined `T` in chunked-prefill loop; latent
  since v7.62.20 ship. Continuous-batching token cap kept us under
  the trigger threshold, so the dead branch was never exercised in
  PROD. Fix above in v7.68 section.
- G-002 — `vllm/_genesis/__init__.py` eagerly imported `prealloc`
  (which imports `torch`) → every torch-less CLI / pre-commit /
  static-analysis tool failed `ModuleNotFoundError`. Fixed via lazy
  `__getattr__` using `importlib.import_module`.
- G-003 + G-004 — `ResponseCacheMiddleware` "never raises / always
  returns response" contract violated two ways: malformed temperature
  string leaked `ValueError` to client as 500; corrupt
  JSON-non-serializable cache entry hung connection because
  `_send_cached_response` returned without sending. Both fixed.
- G-006 — `apply_all_plugins()` ran BEFORE the core patch loop
  despite the docstring saying "After core patches finish, walk
  plugins". Reordered.
- G-007 — `validate_registry()` ran before `register_plugins()`
  injected community entries → plugin entries skipped boot-time
  validation. Re-validation added after plugin apply pass.
- G-008 — 7 env-var references in PATCHES.md / INSTALL.md didn't
  match `dispatcher.py` names. Operators copy-pasting got no-op env
  vars while their patch silently stayed disabled. All 7 synced.

Lint pass (G-016): **195 ruff `F401`/`F841`/`RUF059` errors → 0**.
154 auto-fixes via `--fix`, 41 via `--unsafe-fixes`, 1 manual.

Cleanup (G-005, G-009, G-010, G-011, G-012, G-013, G-014) closed
streaming-doc drift, truncated PATCHES.md row, rig-specific paths
(env-var override + README rationale), Pareto-ranking bug from
unused `speed_runs`, Redis cache size off-by-N for stats keys, PN16
broken doc reference to gitignored internal docs, TextPatcher
Python-only marker now documented.

Pre-commit ruff hook will be added to block this class of latent
bug at commit time instead of waiting for a future audit pass.

---

## v7.65 — 2026-05-02 (repo hygiene + community closeouts)

See top-level [`CHANGELOG.md`](../../CHANGELOG.md) "v7.65" section
for the full list. Highlights: 32 legacy P1–P46 promoted to
first-class PATCH_REGISTRY entries (now 100 total), example pkg
restored, sync gate test for legacy↔registry consistency, boot
validator wired, P67 stale tests fixed. 1470 tests / 0 failures at
the time of cut; v7.68 brought it to 1494 / 0.

---

## v7.63.x — 2026-04-30 (Phase 2.2: repo root cleanup)

The repo root previously had 29 entries — 7 docker-compose files, 2
shell scripts, and a `reference/` dir of upstream PR studies were
mixed in with the canonical top-level docs. Phase 2.2 consolidates
them into purpose-named subdirectories so the root is short and every
top-level item has an obvious purpose.

**Before:** 29 root entries.
**After:** 21 root entries (10 markdown + LICENSE/NOTICE + 1 backwards-
compat shim + 10 directories).

### Moves

- `docker-compose.*.yml` × 7 → `compose/` — relative volume paths
  inside each YAML rewritten `./X` → `../X`. `docker-compose.unit.yml`'s
  `.:/work:ro` (mount cwd) updated to `..:/work:ro` (mount repo root).
- `validate_unit.sh` + `validate_integration.sh` → `scripts/` —
  both gained `cd "$(dirname "$0")/.."` normalization so they work
  from anywhere in the repo. Internal cross-references and compose
  paths updated.
- `reference/` (upstream PR diff studies) → `docs/upstream_refs/` —
  no collision with existing `docs/reference/` (Genesis-internal
  markdown).

### Doc updates

35+ replacements across README.md, INSTALL.md, QUICKSTART.md,
MODELS.md, docs/BENCHMARK_GUIDE.md:

- `docker compose -f docker-compose.X.yml` → `docker compose -f compose/docker-compose.X.yml`
- `./validate_X.sh` → `./scripts/validate_X.sh`
- INSTALL.md "Repository layout" tree rewritten to reflect new
  structure with explicit notes on the three deliberately-kept-at-root
  items.

### Kept at root (intentional)

- `patch_genesis_unified.py` — backwards-compat shim. Multiple
  external repos volume-mount this path in their docker-compose. Its
  own docstring explains why.

### Moved into `tools/` (2026-05-04)

- `genesis_vllm_plugin/` → `tools/genesis_vllm_plugin/` —
  compose mounts updated to `../tools/genesis_vllm_plugin:/plugin:ro`;
  start scripts updated to `/home/sander/genesis-vllm-patches/tools/genesis_vllm_plugin`.
  Repo-root symlink `genesis_vllm_plugin -> tools/genesis_vllm_plugin`
  kept on the deploy server for backwards-compat with any external pinning.
- `external_probe/` → `tools/external_probe/` — referenced from
  CREDITS.md, INSTALL.md, and external compose files (now `../tools/external_probe`).

### Verification

- Session test surface: 469/469 pass (no regressions)
- `genesis self-test --quiet` exits 0
- All 7 compose files retain valid relative paths (verified
  `../vllm/_genesis` and `../tools/genesis_vllm_plugin` exist from within
  `compose/`)

NOT YET DEPLOYED to server — local commit only.

---

## v7.63.x — 2026-04-30 (Phase 2.1: wiring/ disk reorg)

The Phase 2 categories surface (`compat/categories.py`) gave operators
logical grouping — "browse all spec_decode patches", "see everything
in structured_output". But the wiring tree itself stayed flat: 72
`patch_*.py` files in one directory regardless of theme. Phase 2.1
finally lands the disk reorg that was explicitly deferred earlier.

### New layout

```
vllm/_genesis/wiring/
├── text_patch.py              TextPatcher framework + B2 helper
├── rebind.py                  runtime class-method rebind helpers
├── spec_decode/      22 files P56-P79c, P82-83, P86, P94, PN8-9
├── structured_output/ 6 files P59, P61/61b, P62, P64, P68/69
├── perf_hotfix/       4 files P98, P99, P100, P101
├── compile_safety/    3 files P72, P74, P78
├── kv_cache/          2 files P84, P85
├── kernels/           4 files P81, P87, P91, PN14
├── hybrid/            5 files P95, P103, PN11-13
├── middleware/        1 file  PN16 (lazy_reasoner)
└── legacy/           25 files P1-P55 (pre-PATCH_REGISTRY series,
                                apply_all.py dry-run only)
```

15 `compat/categories.py` categories collapsed into 9 disk subdirs:
single-patch categories merged where they share a theme (kernel +
kernel_perf + kernel_safety + quantization → `kernels/`;
cudagraph_safety + memory_hotfix + memory_savings + model_correctness +
stability → `hybrid/`).

### Layout-agnostic resolution

The infrastructure changes shipped in **Phase 2.1a** (commit b66d61e):

- `compat/categories.py:_build_module_index` — `rglob` + computed
  dotted module path. Works for the old flat layout AND the new
  categorical layout transparently.
- `compat/self_test.py:_check_wiring_imports` — same `rglob` walk
  with computed module paths.
- `patches/apply_all.py:_resolve_wiring_module` — new helper that
  caches a `stem → full_dotted_path` index. Both call sites
  (`_wiring_text_patch` and `verify_live_rebinds`) route through it.

### Imports updated

- 60 sites in `patches/apply_all.py`
- 23 test files (~125 imports), including one parenthesized
  multi-name import in `test_wiring_new_patches.py`

### What you'd notice as an operator

Almost nothing — the change is structural, not behavioral.

- `genesis explain P67` still resolves correctly
- `genesis categories --category spec_decode` lists the same patches
- `genesis self-test` still walks all 19 compat modules + all 72
  wiring modules
- Existing recipes / launch scripts unchanged
- All env flags unchanged

What changes:

- New patch contributors should drop their wiring file into the
  matching `<category>/` subdir. `_build_module_index` will pick it
  up via rglob — no registry change beyond the usual `category` field
  in PATCH_REGISTRY.

### Verification

- Session test surface: 469/469 pass (same as pre-reorg)
- Full test directory: 120 pre-existing failures are all
  vllm-not-installed env limitations (test_config_detect, etc.) —
  Phase 2.1 introduced zero new failures
- `compat/categories.module_for(P67)` →
  `'vllm._genesis.wiring.spec_decode.patch_67_tq_multi_query_kernel'`
- `compat/categories.module_for(PN14)` →
  `'vllm._genesis.wiring.kernels.patch_N14_tq_decode_oob_clamp'`

NOT YET DEPLOYED to server — local commit only, awaiting explicit
push approval.

---

## v7.63.x — 2026-04-29 (`genesis recipe diff` for community A/B compare)

The recipe surface was complete for save/load/list/delete/adopt but
missing the one workflow community Q&A actually needs every day:

> "Here's my v794 recipe and yours from the issue thread —
> what's actually different?"

```bash
python3 -m vllm._genesis.compat.cli recipe diff my-prod community-prod
python3 -m vllm._genesis.compat.cli recipe diff a b --json
```

Output is a structured 3-section delta — CHANGED (key + old/new),
REMOVED (only in A), ADDED (only in B) — keyed by dotted paths into
the recipe (e.g. `vllm_serve.max_model_len`).

Provenance fields (`created_at`, `created_by`, `_adopted_from`,
`_adopted_at`, `name`, `description`, `genesis_recipe_version`) are
excluded from the diff so two operators with the same effective config
saved at different times see "identical (no differences ignoring
provenance fields)" rather than meaningless metadata noise.

The new `diff_recipes(a, b)` function is also exported as a library
call for tooling that wants the structured diff without going through
the CLI.

11 new tests pin the contract: identical-clean, added/removed/changed
keys, nested dict handling, list-as-opaque-value semantics, provenance
exclusion, CLI routing, unknown-recipe error code, identical message,
JSON output.

Total session test surface: 456 → 467 passing.

---

## v7.63.x — 2026-04-29 (`genesis bench` joins the unified CLI)

The benchmark suite (`tools/genesis_bench_suite.py`) was previously
reachable only by direct path invocation. It's now a first-class
unified-CLI subcommand:

```bash
python3 -m vllm._genesis.compat.cli bench --quick
python3 -m vllm._genesis.compat.cli bench --mode standard --ctx 8k
python3 -m vllm._genesis.compat.cli bench --compare a.json b.json
python3 -m vllm._genesis.compat.cli bench --ablate-against base.json --ablate-tag no-PN14
```

All argv after `bench` is forwarded verbatim to the underlying script.

The shim (`vllm/_genesis/compat/bench.py`) locates the script via
three candidate paths (env var `GENESIS_REPO_ROOT`, repo-relative,
cwd-relative) so deployments without a source tree can still point at
a checkout. `--help` works even when the script is unreachable —
falls back to a stub pointing at `docs/BENCHMARK_GUIDE.md`.

The bench script's `main()` and `parse_args()` were refactored to
accept an explicit `argv` parameter (was: implicit `sys.argv`). Direct
invocation `python3 tools/genesis_bench_suite.py` is unchanged; this
just removes a side-effect that blocked clean CLI wrapping and unit
tests.

11 new tests (`tests/compat/test_bench.py`) pin: shim importable,
script locator, env override, --help passthrough, slim-deployment
fallback, argv signature contract, parse_args with explicit argv,
unified-CLI wiring, top-level help banner mention.

The `compat.bench` module is now part of self-test's compat-imports
walk: 18 → 19 modules.

Total session test surface: 445 → 456 passing.

---

## v7.63.x — 2026-04-29 (operator tooling: version constant + CI + self-test)

Three small but load-bearing additions on top of the Phase-1..5d compat
layer to make the project operator-friendly and CI-gated.

### NEW canonical version constant

`vllm/_genesis/__version__.py` — single source of truth for the package
version. `vllm/_genesis/__init__.py` re-exports it; telemetry now reads
this constant instead of hardcoding a string. Five tests
(`tests/test_version.py`) pin: importable, format regex, telemetry
consistency, no stale version strings drifting through other modules.

### NEW `.github/workflows/test.yml` — CI gate on every push/PR

Runs the 439-test session suite (compat/* + dispatcher validator + PN14
+ PN16 + B2 wiring helper + D3 ablation bench + version sanity) on a
Python 3.10 + 3.12 matrix. Adds two extra gates:

- `python3 -m vllm._genesis.compat.lifecycle_audit_cli --quiet`
  (exit 1 on unknown lifecycle state)
- `python3 -m vllm._genesis.compat.schema_validator`
  (exit 1 on malformed PATCH_REGISTRY entry)

Genesis stays runtime-dep-free; CI installs only `pytest`,
`pytest-cov`, `packaging`, and a CPU torch build.

### NEW `genesis self-test` — operator-facing structural sanity check

```bash
python3 -m vllm._genesis.compat.cli self-test
python3 -m vllm._genesis.compat.cli self-test --quiet
python3 -m vllm._genesis.compat.cli self-test --json
```

8 checks run after a `git pull` or pin bump:

1. `__version__` constant present + readable
2. All compat modules import cleanly (18 modules)
3. All wiring modules import cleanly (`vllm` may be skipped in slim env)
4. PATCH_REGISTRY validates against schema (no errors)
5. Lifecycle audit clean (no unknown states)
6. Categories index builds + every patch placed
7. Predicates evaluator works on every `applies_to` in the registry
8. Schema file present + parseable (skip in slim deployments)

Different from `doctor`: doctor answers "is my SYSTEM healthy?",
self-test answers "is Genesis itself working?". A doctor failure can
be hardware/config; a self-test failure is a Genesis bug.

The schema-file check (#8) uses multi-candidate path resolution +
`GENESIS_REPO_ROOT` env override so a slim container deployment that
mounts only the package — without the `schemas/` source dir — returns
**skip** rather than fail. Verified inside the live v794 PROD
container: 7 pass, 0 fail, 1 skip, exit 0.

17 tests (`tests/compat/test_self_test.py`) pin the contract — module
import, run shape, individual check name presence, real-registry
all-pass, CLI subcommand routing, JSON output, quiet mode, and two
regression tests for the container path-resolution case.

---

## v7.63.x — 2026-04-30 (Genesis Compat Layer Phases 1 + 4 + Quentin-M cherry-pick)

Phase 1 + Phase 4 in one release. Phase 2 (wiring refactor) and Phase 3
(auto-update channels) deferred to v7.64.x and v7.65.x respectively.

### NEW Phase 4 surface (CLI commands)

- `python3 -m vllm._genesis.compat.explain <patch_id>` — per-patch
  detailed report (identity + lifecycle + dependencies + applies_to
  predicate evaluation + upstream tracker + live `should_apply()`
  decision + credit). Counterpart to `doctor` — doctor shows the
  forest, explain zooms into one tree.
- `python3 -m vllm._genesis.compat.lifecycle_audit_cli` — registry
  audit grouped by lifecycle state. `--state <X>` filter, `--quiet`
  shows only error+warning rows, `--json` for CI. Exit 1 on unknown
  lifecycle state (CI-blocking).
- `python3 -m vllm._genesis.compat.schema_validator` — validates
  every PATCH_REGISTRY entry against
  `schemas/patch_entry.schema.json`. Catches typo'd field names
  (`applys_to`), missing required fields, bad env_flag pattern,
  unknown lifecycle, deprecated-without-supersedes, etc. Exit 1
  on schema violations. Hand-rolled — no `jsonschema` dep required.

### NEW JSON schema for PATCH_REGISTRY entries

`schemas/patch_entry.schema.json` (PEP-style draft-2020-12). Documents
every valid field + lifecycle-conditional requirements (deprecated
must declare `superseded_by` or `deprecation_note`; research must
declare `research_note`; community must declare `community_credit`).
Used by:
- `compat/schema_validator.py` for runtime validation
- IDEs that recognize JSON schemas (auto-completion when editing entries)
- Pre-commit hook (catches typos before commit)

### NEW `genesis recipe adopt URL` — community recipe sharing

Completes the recipe sharing loop. Sander saves his v794 PROD recipe,
pushes the JSON to a public gist or repo, a community user runs:

```bash
python3 -m vllm._genesis.compat.cli recipe adopt \\
    https://gist.githubusercontent.com/.../v794-prod.json \\
    my-prod
```

→ identical launch configuration in one command.

Security model:

- HTTPS-only by default (HTTP refused unless `--allow-http` for testing)
- Body capped at 100 KB by default (recipes are tiny — refuses oversized
  payloads from malicious URLs)
- Schema validated **before** saving (no garbage hits disk)
- Adoption-specific stricter check: rejects "near-empty" recipes that
  pass `validate_recipe` but have no substantive launch info (no
  `target`/`container`/`vllm_serve`/`envs`/`mounts` declared)
- Recipe-name validated by same regex as `recipe save` — no path
  traversal possible
- Provenance: every adopted recipe gets `_adopted_from` + `_adopted_at`
  fields recording the source URL and timestamp

CLI:

```bash
genesis recipe adopt <url> <local-name>
genesis recipe adopt <url> <local-name> --allow-http     # for testing
genesis recipe adopt <url> <local-name> --max-bytes 50000  # tighter cap
```

14 unit tests cover URL validation, body validation, oversized refusal,
adoption persistence, origin tracking, CLI routing.

### NEW reference plugin example — `tools/examples/genesis-plugin-hello-world/`

Working plugin package authors can copy as a starting point:

```text
tools/examples/genesis-plugin-hello-world/
├── pyproject.toml                          # entry-point declaration
├── README.md                               # install + enable + verify
└── genesis_plugin_hello_world/
    ├── __init__.py
    └── plugin.py                           # get_patch_metadata + apply
```

Install:

```bash
pip install -e tools/examples/genesis-plugin-hello-world/
export GENESIS_ALLOW_PLUGINS=1
export GENESIS_ENABLE_HELLO_WORLD=1
python3 -m vllm._genesis.compat.cli plugins list
```

13 unit tests (`tests/compat/test_plugin_example.py`) verify:

- pyproject.toml + module + README all exist at expected paths
- pyproject declares the `vllm_genesis_patches` entry-point
- `get_patch_metadata` returns required fields (patch_id, title,
  env_flag, default_on, community_credit)
- Metadata validates clean against the schema validator
- `env_flag` follows GENESIS_ pattern
- `default_on=False` (good citizenship)
- `apply_callable` resolves via importlib
- `apply()` returns valid (status, reason) tuple
- Full discovery pipeline — example survives `discover_plugins()` with
  lifecycle force-tagged "community" + `_plugin_origin` stamp

This is the pin: if `docs/PLUGINS.md` says "do X to ship a plugin",
the example must actually demonstrate X working end-to-end. Tests fail
if the example drifts from the documented contract.

### NEW unified `genesis` CLI dispatcher

`compat/cli.py` collapses 13 scattered `python3 -m vllm._genesis.compat.X`
invocations into a single entry-point with subcommand routing:

```bash
python3 -m vllm._genesis.compat.cli                  # show all subcommands
python3 -m vllm._genesis.compat.cli doctor
python3 -m vllm._genesis.compat.cli explain PN14
python3 -m vllm._genesis.compat.cli categories --category spec_decode
python3 -m vllm._genesis.compat.cli recipe save my-prod \\
    --from-container vllm-server-mtp-test
python3 -m vllm._genesis.compat.cli plugins list
python3 -m vllm._genesis.compat.cli telemetry status
python3 -m vllm._genesis.compat.cli update-channel check
```

13 subcommands routed:

- `doctor` — system diagnostic
- `explain` — per-patch deep-dive
- `init` — first-run wizard
- `list-models` — model registry browser
- `pull` — HF download
- `lifecycle-audit` — registry lifecycle states
- `validate-schema` — PATCH_REGISTRY shape check
- `categories` — browse by category
- `migrate` — pin-bump runbook
- `recipe` — launch config sharing
- `plugins` — community plugin entry-points
- `telemetry` — anonymized stats
- `update-channel` — apt-style stable/beta/dev

External names use **dashes** (CLI convention); internal modules use
**underscores** — dispatcher does the mapping. Each per-module CLI
keeps working unchanged for backwards compat (operator scripts that
call `python3 -m vllm._genesis.compat.doctor` still work fine).

Exit-code propagation
─────────────────────
Subcommand's exit code is forwarded to the operator. `SystemExit`
from argparse-driven sub-CLIs is caught + propagated cleanly.

Help forwarding
───────────────
`cli doctor --help` calls `doctor.main(["--help"])` (does NOT
intercept at dispatcher level), so each sub-CLI's argparse-generated
help works unchanged.

Test coverage
─────────────
  tests/compat/test_cli.py — 14 tests:
    - main returns int (not None)
    - Unknown subcommand → nonzero
    - --help shows all subcommands
    - KNOWN_SUBCOMMANDS public set covers all 13
    - doctor / explain / categories routing with args
    - lifecycle-audit / validate-schema / list-models / update-channel
      hyphen→underscore aliasing
    - Exit code 2 propagates from sub
    - No-args prints usage banner
    - --help forwarded to subcommand (not intercepted)

  Cumulative session test count: 390 (was 376 before unified CLI).

Live PROD verification
──────────────────────
`docker exec vllm-server-mtp-test python3 -m vllm._genesis.compat.cli
categories --category kernel_safety` runs cleanly — confirms full
pipeline in container.

### NEW Phase 3.x — update channel + check tool

`compat/update_channel.py` adds apt-style update awareness: operators
choose a channel (`stable`/`beta`/`dev`), the tool queries GitHub for
the upstream tip on that channel, compares against local checkout,
and reports if an update is available. **Apply step is intentionally
deferred** — operators using a git checkout can `git pull` manually
based on the check tool's output, which avoids the security and
atomicity risks of building a custom pull machinery.

CLI:

```bash
# Status — current channel + local commit detection
python3 -m vllm._genesis.compat.update_channel status

# Check upstream (24h cached to avoid GitHub API rate-limiting)
python3 -m vllm._genesis.compat.update_channel check
python3 -m vllm._genesis.compat.update_channel check --force-refresh
python3 -m vllm._genesis.compat.update_channel check --json

# Channel management
python3 -m vllm._genesis.compat.update_channel channel get
python3 -m vllm._genesis.compat.update_channel channel set beta

# Apply (currently prints manual git pull instructions)
python3 -m vllm._genesis.compat.update_channel apply
```

Channels:

- **stable** (default) — points at `main` for now (will promote to
  GitHub Releases tag once Sandermage starts cutting tagged releases)
- **beta** — release candidate branch (currently `main` until a
  separate `beta` branch is forked)
- **dev** — `main` HEAD; future-proofed for when a `dev` branch exists

GENESIS_UPDATE_CHANNEL env overrides the persisted choice without
touching disk. GITHUB_TOKEN / GH_TOKEN env, if set, attaches as Bearer
auth to avoid rate-limiting on the GitHub API.

Implementation:
- `_fetch_github_ref(channel)` queries
  `https://api.github.com/repos/Sandermage/genesis-vllm-patches/commits/<ref>`
  via stdlib `urllib.request` — no extra deps.
- 24h result cache at `$GENESIS_UPDATE_DIR/cache.json` with per-channel
  bucketing.
- `detect_local_commit()` uses `git rev-parse --short HEAD` (best-effort).
- Exit codes: 0 up-to-date, 1 update available, 2 error.

Test coverage:

  tests/compat/test_update_channel.py — 18 tests:
    - Default channel = stable
    - Set / get persists across calls
    - set_channel rejects unknown values
    - Env override (GENESIS_UPDATE_CHANNEL) wins over persisted
    - detect_local_commit returns string or None (never raises)
    - check_for_updates returns dict with required keys
    - Network failure handled gracefully (error key set, no crash)
    - 24h cache deduplicates calls
    - --force-refresh bypasses cache
    - update_available correctly computed (match / mismatch / unknown)
    - CLI subcommands (status / check / channel get/set / apply)

  Cumulative session test count: 376 (was 358 before Phase 3.x).

Live PROD verification on v794 container — status command runs cleanly,
shows current channel + storage dir + repo URL.

### NEW Phase 5d — opt-in anonymized telemetry

`compat/telemetry.py` adds a strictly opt-in, PII-free reporting
mechanism to help the Genesis community see "what configs work in
the wild" without collecting anything that could re-identify users.

Two-gate opt-in
───────────────
  GENESIS_ENABLE_TELEMETRY=1     master gate (default OFF)
  GENESIS_TELEMETRY_UPLOAD=1     upload gate (default OFF; deferred)
  GENESIS_TELEMETRY_DIR=<path>   storage override
  GENESIS_TELEMETRY_INCLUDE_PLUGIN_NAMES=1
                                 include plugin patch_ids in reports
                                 (off by default — names could
                                 fingerprint a small group)

What's collected when enabled
─────────────────────────────
- Stable random instance_id (UUID-shaped, persisted locally only)
- Hardware class (rtx_a5000 / rtx_4090 / h100 / ...) — categorical
- Compute capability (sm_86 etc.)
- vllm / torch / triton / cuda / nvidia driver version strings
- Genesis version + commit
- Detected model class + flags (is_hybrid / is_moe / is_turboquant /
  quant_format)
- List of applied core patch IDs (P67, PN14, ...)
- Lifecycle distribution (counts per state)
- Plugin count (NAMES default OFF)
- Run timestamp

What's NEVER collected
──────────────────────
- Hostname / IP / MAC address
- Username / home directory / paths
- Container names / launch script paths
- Env-variable VALUES (only env-FLAG presence implied via patches)
- Specific tokens, model paths, config secrets

Local-first storage
───────────────────
Reports written to `$GENESIS_TELEMETRY_DIR/reports/<timestamp>.json`
(default `~/.genesis/telemetry/reports/`). The `instance_id` lives
at `<dir>/instance_id` with mode 0600. Network upload is **deferred**
until the community dashboard exists — `upload_report()` is a no-op
returning None for now.

CLI
───
```bash
python3 -m vllm._genesis.compat.telemetry status
python3 -m vllm._genesis.compat.telemetry show     # what would be reported
python3 -m vllm._genesis.compat.telemetry collect  # save a report
python3 -m vllm._genesis.compat.telemetry clear    # delete local stash
```

Boot-time integration
─────────────────────
`apply_all.run()` checks `is_enabled()` at end of patch application.
When telemetry is on, an anonymized report is collected + saved
locally. No network call. Off-by-default means zero behavior change
for existing operators.

PII enforcement
───────────────
The test suite includes a strict PII check that scans the entire
report payload as a flat string against:
- Hostname (from socket.gethostname)
- Username (from getpass.getuser)
- Path patterns (/home/, /Users/, /nfs/)
- Env-style strings (=true)

Any leak fails CI immediately.

Test coverage
─────────────
  tests/compat/test_telemetry.py — 23 tests:
    - Default OFF (both gates)
    - Upload requires both master AND upload gate
    - Instance ID stable across calls + persisted to disk + not
      derived from hostname/username
    - Report shape (required keys present, no PII)
    - Patches section: only ID strings, no env values
    - Plugin section: count only by default, names opt-in
    - Storage: timestamped filenames, collision handling, clear
    - Save refused when gate closed
    - Upload: no-op until dashboard launches
    - CLI: status / show / collect / clear

Live PROD verification on v794 container:
  - Status (gate OFF default): correctly reports DISABLED
  - Show with gate ON: produces clean JSON with hardware (a5000 ×2),
    software (vllm 0.20.1+ / torch 2.11 / triton 3.6 / cuda 13.0),
    no PII visible
  - apply_all integration: imports clean, no crash on boot

Cumulative session test count: 358 (was 335 before Phase 5d).

### NEW Phase 5c — `apply_callable` for plugins (plugins can RUN code)

Phase 5b shipped metadata-only plugins. Phase 5c completes the plugin
story by letting plugins ACTUALLY apply their patch — not just declare
it. A plugin now declares:

```python
def get_patch_metadata():
    return {
        "patch_id": "MY_PATCH",
        "title": "My community fix",
        "env_flag": "GENESIS_ENABLE_MY_PATCH",
        "default_on": False,
        "community_credit": "@my_handle",
        "apply_callable": "my_pkg.module:apply",   # NEW
    }
```

Where `apply_callable` may be:

- **A "module:func" string** (preferred — entry-point style, lazy-loaded
  via importlib at boot)
- **A direct Python callable** (advanced — for plugins that compose
  metadata at runtime)

Genesis calls the function during boot when:

1. `GENESIS_ALLOW_PLUGINS=1` (master gate, opt-in)
2. The plugin's `env_flag` is set to a truthy value
3. `apply_all` runs with `apply=True`

### Return value contract

`apply_callable` returns `(status, reason)` where status is one of
`applied` / `skipped` / `failed`. Genesis is forgiving:

- Bare string returned → treated as `("applied", <string>)`
- `None` returned → treated as applied with generic message
- Tuple with invalid status → treated as failed (preserves original status)
- Exception raised → caught, logged, reported as failed (no crash)

This means "lazy" plugins can write `def apply(): return "ok"` and
still play nice with the dispatcher matrix.

### Error isolation

Same guarantees as Phase 5b discovery: one plugin's apply failure
doesn't break others, doesn't propagate out of `apply_all`, doesn't
crash the engine.

### apply_all integration

`apply_all.run()` calls `apply_all_plugins()` after core patches
finish. Stats logged in standard form:

```
[Genesis plugins] apply pass: total=2 applied=1 skipped=1 failed=0
[Genesis plugin] APPLIED MY_PATCH — MY_PATCH applied: did the thing
[Genesis plugin] SKIPPED OTHER_PATCH — opt-in only — set GENESIS_ENABLE_OTHER_PATCH=1
```

### Schema validator

Added `apply_callable` to `_KNOWN_FIELDS` so plugin schema-validates
clean.

### Test coverage

15 new tests in `tests/compat/test_plugin_apply.py`:

- String-form resolution (module:func)
- Direct callable passthrough
- Unknown module / attr / bad format → returns None
- Discovery preserves apply_callable
- Apply: env-set vs env-unset (gate behavior)
- Apply: no apply_callable → skipped (metadata-only)
- Apply: callable raises → failed (error isolation)
- Apply: callable returns garbage → graceful handling
- apply_all_plugins: stats dict shape
- apply_all_plugins: zero when gate closed

Live verification: imports clean inside v794 PROD container.

Cumulative session test count: 335 (was 320 before Phase 5c).

### NEW Phase 5b — community plugin entry-points

`compat/plugins.py` opens Genesis to **third-party patches without
forking the core repo**. A community plugin is just an installable
Python package with one entry-point declaration.

Plugin spec:

```toml
# In a third-party package's pyproject.toml:
[project.entry-points."vllm_genesis_patches"]
my_patch = "my_pkg.patch:get_patch_metadata"
```

Where `get_patch_metadata()` returns a dict (or list of dicts) with
the same shape as a PATCH_REGISTRY entry. See `docs/PLUGINS.md` for
the complete authoring guide.

Genesis enforces:
- **Schema validation** — same checks as core registry (catches typos,
  missing fields, lifecycle-conditional requirements)
- **Lifecycle force** — auto-tagged `lifecycle: community` (plugin
  can't claim "stable")
- **Collision check** — rejected if `patch_id` clashes with core registry
- **Origin stamping** — `_plugin_origin` field shows which entry-point
  the patch came from
- **Failure isolation** — one bad plugin can't break discovery of others

OPT-IN security gate: `GENESIS_ALLOW_PLUGINS=1` to enable. **Default:
zero foreign code loaded.** Genesis boots identically with or without
plugins installed.

CLI:

```bash
GENESIS_ALLOW_PLUGINS=1 python3 -m vllm._genesis.compat.plugins list
GENESIS_ALLOW_PLUGINS=1 python3 -m vllm._genesis.compat.plugins show MY_PATCH
python3 -m vllm._genesis.compat.plugins validate    # works regardless of gate
```

Boot-time integration: `apply_all.run()` calls `register_plugins()` after
the core registry is built but before validators fire. Plugins show up
in:
- `genesis doctor` (with provenance)
- `genesis explain <patch>` (with origin)
- `genesis lifecycle-audit` (under "community" bucket)
- `genesis categories` (in their declared category)
- All standard env-flag gating

18 unit tests + new fields (`patch_id`, `_plugin_origin`) added to
`schema_validator._KNOWN_FIELDS` so the plugin extension protocol is
schema-allowed.

Documentation: `docs/PLUGINS.md` ships with this release — full plugin-
authoring guide including pyproject.toml example, callable shape,
multi-patch packages, security model, etiquette.

### NEW Phase 5a — `genesis recipe` system

`compat/recipes.py` — first-class capture / share / replay of launch
configurations. A "recipe" is a complete reproducible spec for a
Genesis launch (hardware target + container settings + envs +
vllm serve args + expected metrics + notes).

Storage at `$GENESIS_RECIPES_DIR/<name>.json` (default `~/.genesis/
recipes/`), JSON format (no PyYAML dep).

CLI subcommands:
```bash
# Capture a running container as a recipe
genesis recipe save v794-prod --from-container vllm-server-mtp-test \
    --description "27B Lorbus INT4 + TQ k8v4 (v794 PROD)"

# List local recipes
genesis recipe list

# Display one recipe (formatted or --json)
genesis recipe show v794-prod

# Generate a bash launch script from a recipe
genesis recipe load v794-prod --out scripts/launch/start_v794.sh

# Validate a recipe's shape against PATCH_REGISTRY env flags
genesis recipe validate v794-prod

# Delete
genesis recipe delete old-recipe
```

Recipe-name validation: `^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$` — rejects
path traversal, leading dots, separators. Tests verify this.

Live-captured Sander's v794 PROD container as `v794-prod` recipe
(63 envs detected, 5 mounts, all GENESIS_ENABLE_* flags preserved).
Round-trip launch-script generation produces a valid bash script that
recreates the container identically. 25 unit tests, all green.

### NEW `--hf-id-override` for `genesis pull`

Operator can override the registry's `hf_id` per pull. Enables Lorbus
vs Intel quant-variant choice for Qwen3.6-27B INT4 without forking
the model registry:

```bash
# Default registry path: Intel/Qwen3.6-27B-A3B-int4-AutoRound
genesis pull qwen3_6_27b_int4_autoround

# Override to Lorbus's variant (same AutoRound recipe, different quant pass)
genesis pull qwen3_6_27b_int4_autoround \
    --hf-id-override Lorbus/Qwen3.6-27B-int4-AutoRound
```

Smoke-tested live in container with `--dry-run` — pre-flight checks
pass with override path printed.

### NEW Phase 2 surface — categories navigation

`compat/categories.py` provides logical category-grouped navigation
without a physical disk move. Disk reorg deferred to Phase 2.1 — the
flat `wiring/patch_*.py` layout is preserved (zero broken imports).

- `python3 -m vllm._genesis.compat.categories` — list all categories
- `python3 -m vllm._genesis.compat.categories --category spec_decode` — drill in
- `python3 -m vllm._genesis.compat.categories --json` — machine consumer

Programmatic API:
- `category_for(patch_id)` — reverse lookup
- `patches_in(category)` — forward lookup
- `module_for(patch_id)` — resolve to wiring module path
- `import_module_for(patch_id)` — actually load it

48 patches across 15 categories. Categories derived from PATCH_REGISTRY's
`category` field — no manual drift possible. 21 unit tests.

### NEW Phase 3 surface — `genesis migrate-vllm` runbook generator

`compat/migrate.py` is **D1's offline-aware sibling** — D1 runs on CI
cron and reports drift; migrate.py is operator-driven and produces
actionable runbook before a planned pin bump.

Per-patch verdict against an upstream-vllm checkout:

| Status | Meaning | Action |
|---|---|---|
| `clean` | All anchors match exactly once | No action |
| `upstream_merged` | Upstream has merged equivalent fix | Self-retire (none) |
| `anchor_drift` | Anchor not found / refactored | Re-derive anchor |
| `ambiguous_anchor` | Anchor matches >1 times | Narrow anchor with context |
| `file_missing` | Target file moved/renamed | Update wiring path |
| `non_text_patch` | Wiring without _make_patcher | Manual review |

Output:
- Structured dict (JSON) for CI/dashboards
- Human-readable markdown runbook (default `--out`)
- Exit 1 if any patch needs operator action

Usage:

```bash
# Quick: clone target vllm + run runbook
git clone --depth 1 -b v0.21.0 https://github.com/vllm-project/vllm.git /tmp/v021
python3 -m vllm._genesis.compat.migrate /tmp/v021 --out runbook.md
```

13 unit tests using synthetic vllm-tree fixtures (no real upstream
clone needed for CI).

### NEW pre-commit hook for contributors

- `scripts/git/pre-commit` — bash hook that runs schema validator +
  A3/D2 dispatcher validator + lifecycle audit on every commit
  affecting PATCH_REGISTRY-relevant files (regex match on changed
  paths). Fast-pass on doc-only commits.
- `scripts/git/install.sh` — one-command symlink installer.
- Skip with `git commit --no-verify` (use sparingly).

### Quentin-M P67b cudaErrorIllegalAddress fix (cherry-pick)

Cherry-picked from `Quentin-M/genesis-vllm-patches` branch
`fix_p67b_illegal` (not yet PR'd to Sandermage main — found via
cross-fork survey). Replaces shared `buf_holder=layer` in the P67b
upstream path with a per-K1 `SimpleNamespace` holder on `self`. Defensive
against any model where Hq/Hk is non power-of-2 (e.g. Qwen3.6-27B with
Hq/Hk=5, where the custom P67 Triton kernel can't compile and falls
back to the upstream path which had OOB write on B×K1 synthetic rows).
Fresh-container restart on v794 PROD verified clean — tool-call 4/4,
HTTP 200, no regressions. Marker bumped: `v7.62.12_baked_env` →
`v7.63.x_quentin_buf_holder_fix`. Tracked in `upstream_compat.py` as
`QUENTIN_M_P67b_BUF_HOLDER_FIX`.

### Community fork research findings

Surveyed 8 contributor repos (JartX, Quentin-M, noonghunna, thc1006,
MidasMining, jhsmith409, webcodes-cz, ampersandru). Highlights:

- **noonghunna/club-3090** (★85), `qwen36-27b-single-3090` (★78),
  `qwen36-dual-3090` (★40) — multi-engine recipes for Qwen3.6-27B on
  RTX 3090. Their **Cliff 1+2 documentation more granular than ours**;
  Cliff 2 (DeltaNet GLA recurrent state buffer overflow on >50K single
  prompts) is new info — investigate in v7.64+ if PN12-style pooling
  could help GLA path.
- **thc1006/qwen3.6-speculative-decoding-rtx3090** (★14) — empirical
  finding: *"no llama.cpp spec-decode variant achieves net speedup on
  Ampere + A3B MoE"*. Cross-engine counter-data; vllm MTP K=3 still
  wins on Genesis PROD.
- **JartX has 5 OPEN vllm PRs** in the per-token-head INT2/INT4 family.
  Tracked in `upstream_compat.py` as `PR_40835_*`, `PR_39939_*`,
  `PR_39074_*`. **Warning:** #39939 will require Genesis anchor
  re-derivation for PN14 + P40 when merged.
- **jhsmith409/llama-cpp-turboquant-gemma4** — TQ ports to llama.cpp for
  Gemma 4 D=256/512. Cross-engine reference for v8.x arena work.

Full details in `memory/project_genesis_community_research_20260430.md`
(internal).

### NEW Phase 1 surface (recap, shipped earlier this release)

- `python3 -m vllm._genesis.compat.doctor` — diagnostic CLI
- `python3 -m vllm._genesis.compat.init_wizard` — first-run wizard
- `python3 -m vllm._genesis.compat.models.list_cli` — model browser
- `python3 -m vllm._genesis.compat.models.pull <key>` — HF download
- New `compat/` module: predicates, version_check, lifecycle, fingerprints,
  models registry. Re-export shims preserve legacy import paths.
- Richer `applies_to` DSL with AND/OR/NOT/NONE_OF compounds.
- Reference fingerprint: `rtx_a5000_x2_qwen3_6_27b_int4_v794.json`.

### Test coverage (Phase 1 + 4 cumulative)

| Suite | Tests |
|---|---:|
| `tests/compat/test_predicates.py` | 33 |
| `tests/compat/test_version_check.py` | 20 |
| `tests/compat/test_lifecycle.py` | 13 |
| `tests/compat/test_models_registry.py` | 11 |
| `tests/compat/test_doctor_smoke.py` | 6 |
| `tests/compat/test_explain.py` | 24 |
| `tests/compat/test_lifecycle_audit_cli.py` | 9 |
| `tests/compat/test_schema_validator.py` | 15 |
| Pre-existing v7.63.x test suites (validator/PN14/PN16/B2/D3) | 112 |
| **Total session test count** | **243** |

### Live validation

Doctor + explain + schema_validator all run cleanly inside v794 PROD
container (Lorbus 27B INT4, 27 patches APPLY, 0 schema issues). Tool-call
4/4, HTTP 200, container uptime preserved during sync.

### Phase roadmap (updated)

- ✅ **Phase 1** — compat module + doctor + models + predicates + lifecycle
- ✅ **Phase 4** — explain CLI + lifecycle-audit + JSON schema + pre-commit
  (ahead of original sequence; was originally after Phase 2)
- 🚧 **Phase 2** — wiring refactor into category subdirs (next sprint)
- 📋 **Phase 3** — auto-update channels (stable/beta/dev) + migration runbook
- 📋 **Phase 5** — plugin entry-points + opt-in telemetry + recipes

---

## v7.63.x Phase 1 (initial release) — see entry above merged into combined section.

## v7.62.x — 2026-04-29 (PN12 + PN14 + PN16 + A3/D2 validator + B2/D1/D3/D4)

Major architectural milestone. Turns Genesis from a "patcher running on
Sander's machine" into a discoverable, self-documenting, hardware-aware
product surface. Backward-compatible with all v7.62.x configs — no PROD
disruption.

### New surface (CLI commands)

- `python3 -m vllm._genesis.compat.doctor` — single-command diagnostic
  (hw + sw + model + patches + lifecycle + validator). Human-readable
  or `--json` for CI.
- `python3 -m vllm._genesis.compat.init_wizard` — interactive first-run
  setup (detect hw → recommend model → workload pick → generate launch
  script).
- `python3 -m vllm._genesis.compat.models.list_cli` — browse the curated
  model registry.
- `python3 -m vllm._genesis.compat.models.pull <key>` — one-command HF
  download + verify + tailored launch script generator.

### New `compat/` module — single home for all detection / UX

```text
vllm/_genesis/compat/
├── doctor.py             — diagnostic CLI
├── init_wizard.py        — first-run wizard
├── version_check.py      — vllm/torch/cuda/triton/driver range matching
├── predicates.py         — AND/OR/NOT applies_to evaluator
├── lifecycle.py          — patch lifecycle state machine
├── gpu_profile.py        — re-export shim (legacy import path preserved)
├── model_detect.py       — re-export shim
├── config_detect.py      — re-export shim
├── models/
│   ├── registry.py       — SUPPORTED_MODELS dict (5 entries)
│   ├── pull.py           — HF download + verify + launch script gen
│   └── list_cli.py
└── fingerprints/
    └── rtx_a5000_x2_qwen3_6_27b_int4_v794.json
```

### Engineering changes

- **Richer `applies_to` predicate DSL** — AND/OR/NOT/NONE_OF compound
  forms. Solves "INT4 alone doesn't need this, INT4+TQ does". Backwards-
  compatible with all 48 existing flat-dict entries (auto-normalized
  via `predicates.normalize_legacy_rule`).
- **Version-range matching** — patches can now declare `vllm_version_range`,
  `torch_version_min`, `triton_version_min`, `cuda_runtime_min`,
  `nvidia_driver_min`, `compute_capability_min`/`compute_capability_max`.
  Validator at boot enforces. Conservative pass on detection failures.
- **Patch lifecycle states** — `experimental` / `stable` / `deprecated` /
  `research` / `community` / `retired`. Code removal requires prior
  `lifecycle: retired`. Doctor surfaces deprecation `superseded_by`
  actionably.
- **Reference fingerprints** — first entry is `rtx_a5000_x2_qwen3_6_27b_int4_v794`:
  103.3 TPS (CV 4.9%, n=20), 9.36 ms TPOT, 127 ms TTFT, 21964/24564 MiB
  per rank, 256K context verified, tool-call 100%.
- **Curated model registry** — 5 entries with full metadata: HF id,
  size, quant, model_class, hybrid/MoE flags, min VRAM per TP rank,
  tested hardware classes, blessed launch configs, expected metrics,
  license, gating, known quirks, lifecycle status.
- **Dispatcher integration** — `_check_applies_to()` extended to
  delegate to `compat.predicates.evaluate` when compound forms detected,
  and to `compat.version_check.check_version_constraints` for version
  keys. Legacy flat-dict path unchanged.

### Test coverage (new in v7.63.x)

- `tests/compat/test_predicates.py` — 33 tests
- `tests/compat/test_version_check.py` — 20 tests
- `tests/compat/test_lifecycle.py` — 13 tests
- `tests/compat/test_models_registry.py` — 11 tests
- `tests/compat/test_doctor_smoke.py` — 6 tests

**Total new compat tests: 84, all green.**

### Phase roadmap

- ✅ Phase 1 (this release) — compat module + doctor + models + version
- 🚧 Phase 2 — refactor `wiring/patch_*.py` into category subdirs
- 📋 Phase 3 — auto-update channel system (`stable`/`beta`/`dev`)
- 📋 Phase 4 — pre-commit + JSON schema + `genesis explain`
- 📋 Phase 5 — plugin entry-points + opt-in telemetry + recipe system

---

## v7.62.x — 2026-04-29 (PN12 + PN14 + PN16 + A3/D2 validator + B2/D1/D3/D4)

A focused 36-hour optimization sprint that turned into a much bigger
story than expected. Detailed timeline in [README.md §What's new in
v7.62.x](../../README.md#whats-new-in-v762x--36-hour-session-timeline).

### Patches added

- **PN12** — FFN intermediate scratch pool (Cliff 1 fix on TQ3 path).
  Closes 138 MiB OOM at 192K + tool-call. Cross-engine inspiration:
  TensorRT-LLM live-range arena.
- **PN13** — CUDAGraphWrapper `gc.collect`/`empty_cache` lambda arity
  fix (vllm#41235 backport). Defensive vs Blackwell.
- **PN14** — TQ decode IOOB `safe_page_idx` clamp (vllm#40074 backport).
  Defensive vs Triton bounds-checker on >32K sequences.
- **PN16** — Lazy reasoner request hook. Hybrid policy: variant 1
  (pre-decision) + variant 3 (client override) + variant 5 (prompt-
  engineering soft cap). Variant 4 (LogitsProcessor cap) upstream-
  blocked when speculative_config is set; documented in
  `docs/_internal/PN16_PHASE2_UPSTREAM_BLOCKER.md`.

### Infrastructure additions

- **A3/D2 PATCH_REGISTRY validator** — `requires_patches` /
  `conflicts_with` declarations, boot-time validation, caught 2 real
  prod-config issues at first run.
- **A4 P71 hardening** — 6 defensive guards on block_verify_sampler
  launcher (fail LOUD vs silent corruption).
- **B1 buffer-sizing audit** — clean (no NemotronH-class bugs found).
- **B2 `result_to_wiring_status` helper** — DRY across 5 PN-family
  wiring modules; caught silent SKIPPED-as-APPLIED bug class.
- **D1 CI drift watcher** — `tools/check_upstream_drift.py` +
  `.github/workflows/upstream_drift_watcher.yml`. Daily check.
- **D3 bench ablation mode** — `--ablate-against` flag for
  `genesis_bench_suite.py`. Welch t-test + per-metric delta table.
- **D4 external_probe migration** — README + deprecation notices on
  redundant probes.
- **C1 TRT-LLM arena design (v8.x)** — internal design doc filed for
  Phase 2+ work.

### Production validation

- v794 promoted to PROD 2026-04-29: 2× A5000 + 27B Lorbus INT4 + TQ
  k8v4 + 5 PN-family patches. **+17% TPS** vs v771b baseline (88 → 103).
- 256K context verified, 121s prefill, tool-call 7/7 + 4/4 cities.
- VRAM headroom: 2.6 GB per rank (21964/24564 MiB).

---

## v7.59 — 2026-04-28 (PROD promotion: 320K context + smaller batched-tokens)

After overnight investigation closed v756 stability (P67 safety gate fix in
v7.56) Sander asked for "220K+ context to calmly hold the limit". Tested
`--max-model-len 320000` + `--max-num-batched-tokens 4096` against PROD
v7.52 baseline.

**Validated as strict upgrade:**

- Boot: clean, both A5000s healthy, +462-482 MiB MORE free VRAM at boot
  (smaller batched-tokens saves more than larger context costs)
- Long-context probes (think-OFF mode):
  - 280K: 94.0s ✓ (was 400 over old 256K limit)
  - 300K: 105s ✓
  - 317K: 115s ✓ (just under new 320K hard limit)
- Speed bench: 244 → 200 t/s @ max_tokens 64 → 2048 (matches v7.52 class)
- Stability 30 sequential: **30/30** @ 215 t/s, CV 6.84% (was 6.44%)
- Stress 30 (3 concurrent × 10 burst): **30/30** @ 231 t/s, CV 6.67%
  (was 6.99%)
- Both think-ON and think-OFF modes equally fast at long context
- 5 short smoke + 1 280K + 1 300K probes on PROD all OK after promotion

**Promoted to PROD 2026-04-28 23:53 UTC.** Launch script:
`/home/sander/launch_scripts/current/start_v759_320k_prod.sh`. Old v7.52
PROD launch archived as `start_v748_p82_prev_prod_archived_20260428.sh`.

**Knob changes (only THREE from v7.52 PROD):**

- `--max-model-len`: 262144 → 320000
- `--max-num-batched-tokens`: 8192 → 4096
- `GENESIS_TQ_MAX_MODEL_LEN` env: 262144 → 320000 (P37 cap match)

Everything else identical: TQ k8v4, MTP K=3, P67 (with v7.56 safety gate),
P82 t=0.3, all 43 Genesis patches. CONFIGURATION.md "Tested baseline"
updated. README badge bumped (160-190 → 200-244 t/s; +context-320K badge).

## v7.56.2 — 2026-04-27 night (P75 wiring fix scoped import os)

P75 text-patch injected `os.environ.get(...)` into `vllm/config/speculative.py`
`__post_init__` but that file does NOT import `os` at module level → first
v758 boot died with NameError before vllm config was constructed.

Fix: `import os as _genesis_p75_os` scoped to the injected block, all
references switched to `_genesis_p75_os.environ.get(...)`. Marker bumped
to `v7.56_local_os_import` so patch re-applies cleanly on fresh containers.

v758 P75 Suffix Decoding deploy variant TESTED 2026-04-27 23:03 UTC after
fix:
- Boot succeeds
- P75 swap `ngram → suffix` confirmed in logs
- Speed bench INCONCLUSIVE (high variance on standard bench with varied
  prompts; suffix decoding needs real-workload bench with prompt repetition)
- Decision: NOT promoted; kept as opt-in variant for operators with
  agentic/tool-call workloads

## v7.56.1 — 2026-04-27 night (P67 safety gate fix: argv probe)

`config_detect.get_runtime_profile()` short-circuits with
`spec_decode_enabled=False` when `vllm_config` is unavailable (apply_all
phase ALWAYS hits this path). My v7.56 P67 safety gate triggered on PROD
v748 because it couldn't detect the `--speculative-config` from launch
args at apply_all time → P67 SKIPPED on PROD → -32% TPS (lost the 160+
tok/s).

Fix: `_probe_spec_decode_from_argv()` fallback that reads:
1. `GENESIS_FORCE_SPEC_DECODE` env (operator escape hatch)
2. own `sys.argv` (covers `python -m vllm` direct calls)
3. `/proc/1/cmdline` (covers Docker entrypoint where `bash -c "...;
   vllm serve"` is PID 1 at apply_all time before `exec vllm serve`)

Also: P67b mirror gate added (`patch_67b_spec_verify_routing.py::apply()`).
Without P67 active, P67b's forward() routing would dispatch into a
disabled kernel for any non-decode batch → reproducing the v756 IndexKernel
overflow.

Validated:
- PROD v748 (has `--speculative-config` in PID 1 cmdline): P67 APPLY,
  P67b APPLY → +32% TPS preserved.
- v756-ascetic (no `--speculative-config`): P67 SKIP via gate, P67b SKIP
  via gate → bench passes 50/50 + 150/150.

## v7.56 — 2026-04-27 night (P67 safety gate, root cause of v756 crash)

Bisect cleanly identified Genesis P67 multi-query kernel hook as the root
cause of v756 sustained-load crash:

- v756 / v756-ascetic: crash IMA at burst 11-21
- B3-alt-2 (vanilla nightly + auto kv): PASS — bug NOT purely upstream
- B4 (Genesis ascetic + auto kv): PASS — TQ k8v4 part of trigger
- B2 (CUDA_LAUNCH_BLOCKING=1): EXACT line captured →
  `gpu_model_runner.py:4099 hidden_states[logits_indices]`
- **B5 (P67=0): PASS** — root cause confirmed
- v756 + safety gate (P67=1 in env, gate auto-disables): PASS

Fix shipped:
1. `vllm/_genesis/config_detect.py::recommend("P67")` returns
   `"skip:no speculative_config"` when spec-decode is not configured
2. `vllm/_genesis/wiring/patch_67_tq_multi_query_kernel.py::apply()`
   SAFETY GATE: refuses to apply even when env flag is set if
   config_detect verdict starts with "skip"
3. Operator override via `GENESIS_FORCE_SPEC_DECODE=mtp` if needed

PROD v748 (has spec-decode): unaffected — P67 still APPLY → +32% TPS
preserved.
v756-style deploys (cache ON, no spec-decode): now safe under sustained
burst — P67 auto-skipped by safety gate.

NO upstream issue posted — Genesis-side bug, not vLLM. Per Sander rule
"не пишем в ошибки без точных данных и перепроверок с тестами",
investigation completed all narrowing before any draft.
PATCHES.md updated with P83-P86 entries that were missing.

## v7.55 — 2026-04-27 night (v756 stability investigation + bisect)

Sander green-lit v756 (cache ON + align + no spec-decode) live
reproducer + bisect during overnight session. Outcome:

- **v756 sustained-load crash REPRODUCED** at burst 21/30. Root error:
  `pytorch IndexKernel.cu:111 index out of bounds`, async-surfacing at
  `gpu_input_batch.py:1013 update_async_output_token_ids` →
  NVIDIA Xid 43 on both A5000s. Saved: `docs/reference/v756_crash_20260427.log`.

- **Bisect 1 — v756-ascetic** (P83/P84/P85=0): IDENTICAL crash. Genesis
  cache patches NOT the cause. `docs/reference/v756_ascetic_crash_20260427.log`.

- **Bisect 2 — B3 vanilla nightly** (zero Genesis patches, but
  `--kv-cache-dtype turboquant_k8v4`): didn't even boot —
  `NotImplementedError: TurboQuant KV cache is not supported for hybrid
  models`. Vanilla rejects TQ k8v4 + hybrid; Genesis P4 lifts that.

- **Bisect 3 — B3-alt-2 vanilla + auto kv** (no TQ, no Genesis):
  **PASSED ALL TESTS**. 50/50 stability + 150/150 stress + 134-138 tok/s
  clean. → `--kv-cache-dtype turboquant_k8v4` is part of the trigger.

- **Bisect 4 — B4 Genesis-ascetic + auto kv** (Genesis text-patches active
  but NO TurboQuant): in progress at v7.55 cut time. If passes →
  TurboQuant k8v4 + cache + chunked-prefill + sustained burst is the
  single trigger. If crashes → some non-cache Genesis patch is involved
  too.

- **Per Sander rule 2026-04-27 night** ("только нечего не пишем в ошибки
  без точных данных и перепроверок с тестами") — NO upstream issue
  drafted yet. Triple-confirm narrowing continues until either a clean
  upstream-only repro is captured OR Genesis-side cause is localized.

- **Memory rule added** to `feedback_github_comment_style.md`: "no
  upstream posts without exact data + retest verification" with
  acceptance criteria.

- **PATCHES.md** updated to include P83, P84, P85, P86 rows that were
  missing (added 2026-04-27 cycles).

- **PROD v748** untouched throughout. Total downtime to date: ~25 min
  across 4 swap windows (v756 → v748 → v756-ascetic → v748 →
  B3-vanilla → v748 → B3-alt-2 → v748 → B4).

## v7.54 — 2026-04-27 (Quick wins: P86 + bench v4 rename + 2 deferred research files)

Continuation of v7.53 community-engagement round. Sander approved
"Quick wins (часы, low risk) тут 1,2,3,4 эти пункты реализовываем"
plan. Outcome:

- **P86 (vllm#40876 backport) — IMPLEMENTED, opt-in.** New wiring
  `vllm/_genesis/wiring/patch_86_ngram_batch_propose_linear.py`.
  Replaces O(N\*K) `i in valid_ngram_requests` membership scan in
  `NgramProposer.batch_propose` with O(N+K) direct-fill loop. Both
  anchors verified unique on upstream `ngram_proposer.py` (lines 87 +
  121). Default OFF via `GENESIS_ENABLE_P86=1`. Negligible at Genesis
  prod max\_num\_seqs=2 (~ns); meaningful at high-concurrency multi-
  user serving (N=64/K=32 saves ~1952 list-membership ops/batch).
  Wired into Dispatcher v2 PATCH\_REGISTRY + apply\_all `@register_patch`.
  Synced to server `/home/sander/genesis-vllm-patches/`. AST OK +
  dispatcher dry-run shows `P86 SKIP opt-in only — set
  GENESIS_ENABLE_P86=1 to engage` as expected.

- **`scripts/genesis_bench_v4.py` — synced from server.** Byte-
  identical to local `genesis_bench_v3.py` (the v4 fix had already been
  backported to v3 in v7.46). v4 now the canonical name; v3 retained as
  alias until Genesis v8.0. README updated.

- **2 deferred research files written** (so we don't re-investigate):
  - `docs/reference/DEFERRED_P50_DEPLOY.md` — cliproxyapi P50
    middleware deployment path (drop into `genesis-proxy:8318` as
    FastAPI middleware; cliproxyapi itself is a Go binary, no plugin
    surface). Pickup checklist included.
  - `docs/reference/DEFERRED_P87_PR40924.md` — `merge_attn_states_kernel`
    SM-shared-mem (PR #40924) is a CUDA `.cu` patch, not Python — not
    backportable via Genesis text-patch. Wait for upstream merge → bump
    pin → land for free. Workload-relevance check: kernel is off
    Genesis PROD's hot path anyway (no prefix-cache).

- **Status:** zero kernel/runtime change for current PROD; P86 is
  opt-in research; bench v4 alias is no-op rename. PROD remains v748
  (cache OFF + MTP K=3 + P82 t=0.3) — unchanged.

## v7.53 — 2026-04-27 (Research sprint: Tier 3 I REJECTED, P82 SGLang acceptance IMPLEMENTED (opt-in, unvalidated))

This is a research-only delta (no kernel/wiring code change pending Sander's
GO). It captures TWO investigations triggered by the v7.52 follow-up sprint
so future agents (and future-Sander) don't re-spelunk.

### Tier 3 I — vllm#38786 "Splitting MLA attention Triton kernel" — REJECTED

**Source:** [vllm#38786](https://github.com/vllm-project/vllm/pull/38786) — adds a 2D split
(KV_SPLITS × TEMP_PAGES=32) to the **MLA grouped decode kernel**, with a
stage-2 reduce of partial `(acc, e_sum, e_max)` triples back into `att_out`
via online-softmax rescale.

**Why we evaluated it:** carried forward from v7.52 sprint as the highest-EV
remaining throughput candidate. Memory budget claim (`+67 MB scratchpad on
200K context`) checks out: `temp_pages=32 × B=1 × H=128 × KV_SPLITS=8 × (Dv=512+2) × 4 B ≈ 67.4 MB`.

**Why we will NOT backport:**

1. **Wrong kernel family.** The PR modifies `vllm/v1/attention/ops/triton_decode_attention.py`
   (MLA grouped decode, head_dim=512, K+1=1 single-query). Our P67 is GQA
   multi-query spec-decode verify (Qwen3.6-MoE, head_dim=128, K+1=4).
   The split-KV × temp_pages reduction structure does not map: P67 is
   already split-M, single-pass, and the `(B, num_kv_heads)` grid already
   saturates the 64 SMs of an A5000 at modest seq-len. Adding a TEMP_PAGES
   third axis would multiply launch count 32× with no occupancy gain.
2. **Author already disowned it.** After PR #33529 changed the
   `num_kv_splits` calculation (which we adopted in our v7.50 Step C),
   the patch makes throughput *worse* on current main. The PR is OPEN
   with merge conflicts and effectively superseded.
3. **Workload mismatch.** Targets `batch < 32` long-context decode (GLM-4.7-Flash,
   DeepSeek-V3 on TRITON_MLA backend). Our spec-decode is `batch=1, K+1=4`
   short queries on ≤32k context.
4. **Likely register spill on Ampere.** No autotune in the PR; defaults
   `BLOCK_H=16, num_warps=4, BLOCK_DV=512` → acc tile ~32 KB virtual on
   our SM 8.6 — same spill pattern that killed our v7.52 fused-M attempt.

**Reviewer flags also caught:** missing FP8 dequant on `kpe`, OOM at large batch,
NaN risk on empty KV split (gemini-code-assist).

**Decision:** documented and dropped. No kernel work, no opt-in env-flag —
the patch is wrong for our architecture, full stop. If a future agent is
again tempted to investigate split-KV for P67, the right move is to look
at FlashDecoding 2.0 (split-K + atomic reduce) targeted at GQA, not the
MLA-specific TEMP_PAGES design from #38786.

### P82 — SGLang threshold_single OR-clause acceptance (implemented opt-in, NOT validated)

**Source:** SGLang `sgl-kernel/csrc/speculative/speculative_sampling.cuh`
(roughly line 107):

```cpp
if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
    accept;
}
```

vs vLLM's vanilla rule in `vllm/v1/sample/rejection_sampler.py:797`:

```python
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

**Why this matters for us:** our v7.13 strict-ngram analysis identified the
structural ceiling `clean_rate ≈ accept_rate^num_spec`. SGLang's OR-clause
short-circuits whenever the target is even moderately confident
(`target_prob >= threshold_single`), which decays the exponent slowly and
breaks the ceiling. For greedy / low-temp tool-call workloads (our case),
the bias is in the right direction (toward higher-prob target tokens).

**Catch:** the threshold rule is biased — it loses the unbiased-sampling
guarantee of canonical rejection sampling. SGLang accepts this trade-off
explicitly in their docs. For temperature-0.7 tool-call output, expected
quality impact is small but MUST be empirically validated.

**Implementation plan (P82 written — wiring + dispatcher + apply_all hooks present, default OFF):**

1. Text-patch `vllm/v1/sample/rejection_sampler.py` line ~797:

   ```python
   # Genesis P82 anchor:
   accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
   # → replaced (when GENESIS_ENABLE_P82=1) with:
   accepted = (
       (draft_prob > 0 and target_prob / draft_prob >= uniform_prob)
       or (target_prob >= GENESIS_P82_THRESHOLD_SINGLE_LITERAL)
   )
   ```

   Threshold baked as a fp32 literal at apply() time from env
   `GENESIS_P82_THRESHOLD_SINGLE` (default 0.3).
2. Drift-detect markers (P71-style): catch upstream rewrites of the line.
3. Eligibility: ONLY random-sample path (greedy already accepts on argmax
   match — threshold_single doesn't apply).
4. Fallback: any patch failure → upstream untouched, server still boots.
5. Dispatcher entry: `default_on=False`, `category=spec_decode`, `credit=SGLang`.

**Test plan (must run before any prod deploy):**

| Threshold | Expected accept rate | Expected quality | Risk |
|---|---|---|---|
| 0.0 (vanilla) | baseline | 30/31 | none — equiv to OFF |
| 0.2 | +5-10% accept | 30/31 expected | low |
| 0.3 (SGLang typical) | +10-20% accept | 29-30/31 expected | medium |
| 0.5 | +20-30% accept | 27-29/31 expected | high |

Sweep on prod via blue/green container with `genesis_quality_harness.py`
(needle, math, code, tool-call subsets) + `genesis_bench_v3.py` (TPS sweep).
**SHIP CRITERIA:** ≥30/31 quality preserved AND ≥+10% effective TPS at
chosen threshold. ANY quality regression → reject.

**Empirical SGLang baselines (sparse, public):** EAGLE3 on Llama-3-70B
average accepted length ~3.5/7 (~50% per-token) per LMSYS SpecForge blog
2025-07-25. No SGLang Qwen3-MoE numbers published. Our v7.13 strict-ngram
baseline is ~96% (multi-query diverse) / 100% (single-query) clean rate.
P82 target: maintain or improve clean rate while raising effective TPS.

### Other cleanup this delta

- `dispatcher.py`: P56 and P57 now have explicit `deprecation_note` field
  (parity with P63). Registry entries kept (tests rely on them); wiring
  files unchanged. Cosmetic only.

### NOT pushed

Per Sander's standing rule (2026-04-27): no push to `origin/main` without
100% win on full sweep + explicit "ok push". This entire delta is local
commits only (tag `pre-tier-3-i-2026-04-27` for rollback baseline).

---

## v7.52 — 2026-04-27 (Tier 3 H: fused-M kernel as opt-in; REJECTED for prod default)

Implemented Tier 3 H from the throughput sprint plan: fused-M variant of
P67 multi-query kernel. Adapted from the FP64 reference impl in private
repo `Sandermage/p67-genesis-kernel/p67_dev/p67_test_ieee_precision.py`,
ported to our production kernel signature with all v7.50/v7.51 opts
(tl.exp2 + LOG2E, -FLT_MAX, cache_modifier hints, tl.range, hoisted
invariants).

### Added

- `_build_kernel_fused()` in `p67_multi_query_kernel.py` — opt-in via
  env `GENESIS_P67_USE_FUSED=1`. Same kernel signature as split-M for
  caller compat. Architecture: ONE dot per KV-tile with
  `m=K_PLUS_1*HEADS_PER_KV=32`, vectorized online softmax over BLOCK_M
  rows with per-row causal mask `q_abs_pos[:, None] >= seq_offset[None, :]`
  (this is the v7.27 drift fix — finally validated).

### Empirical (validated 2026-04-27, opt-in test on prod)

| Metric | v7.51 split-M | v7.52 fused | Δ |
|---|---|---|---|
| @ 64 tok | 191.0 | 185.0 | -3.1% |
| @ 128 tok | 172.5 | 161.7 | -6.3% |
| **@ 256 tok** | 160.1 | 134.0 | **-16.3%** |
| @ 512 tok | 144.9 | 134.7 | -7.0% |
| @ 1024 tok | 132.0 | 128.4 | -2.7% |
| @ 2048 tok | 137.7 | 134.1 | -2.6% |
| **Stability mean** | **167.2** | **155.5** | **-7.0%** |
| Quality 30-shot | 30/31 | **30/31** | preserved |
| Tool-call 2/2 | PASS | **PASS** | preserved |

### Verdict: KEEP DEFAULT split-M, retain fused as opt-in

Quality preserved (no numerical drift) → **the v7.27 per-row online
softmax fix actually works**. The drift problem that led us to split-M
in v7.34 is solved by `q_abs_pos[:, None] >= seq_offset[None, :]`
broadcast (per-row causal mask). This is genuinely useful knowledge
captured in working code.

But throughput regressed by 7-16% because of register spill:
the fused `acc` tensor is `[BLOCK_M=32, BLOCK_D=128]` fp32 =
**16 KB virtual registers per CTA**, exceeding the A5000 register
budget (64 KB per SM, shared across active warps). Triton compiler
spills to local memory, and each `acc * alpha + dot(P, V)` then goes
through L1/L2 — far slower than the theoretical MMA-count savings
from fewer dots.

This is the same pattern as Steps E (num_stages 3→2, rejected) and F
(Q hoist, rejected): **theoretical optimization wins on consumer Ampere
require respecting register pressure**. With 64 KB per SM (RTX A5000)
and our acc tensor demand, fused-M is in the spill regime.

### Why we keep the fused code in source

1. **Reference for future maintainers** — when someone considers
   fused-M again, the rejection rationale + working code prevents
   re-treading.
2. **Useful on different hardware** — A100/H100 (96 KB / 228 KB per
   SM) may have enough register file. Operator can opt in to test.
3. **Useful at smaller BLOCK_D** — if we ever serve a model with
   HEAD_DIM=64 (not 128), `acc` shrinks to 8 KB; fused may win there.
4. **Opt-in via `GENESIS_P67_USE_FUSED=1` is zero-cost when off** —
   `_build_kernel()` checks env at module load, returns split-M if
   not set.

### Snapshot tag for rollback

`pre-tier-3-h-2026-04-27` (set before this commit).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.51.2 — 2026-04-27 (cleanup pass: torch.cat → slice-assign in fallback paths + bench v3 fix)

Pure-cleanliness pass — no behaviour change in our hot path, no measurable
performance delta (within ±5 tok/s noise band of v7.51 baseline 167 mean).
Quality 30/31 preserved.

### Changed

- **`vllm/_genesis/wiring/patch_38_tq_continuation_memory.py`** — fallback
  path (fires only when prealloc pool isn't wired, e.g. AMD/CPU tests):
  replaced final `torch.cat([k_cached_trim, key_chunk])` with the same
  `pre-allocate-then-slice` idiom used in our main `use_persistent` branch.
  Eliminates one allocation peak in the rare fallback path. Behaviour-
  equivalent.

- **`vllm/_genesis/kernels/block_verify_sampler.py`** — both `cu_start`
  construction sites (lines 131 + 275) replaced `torch.cat` with
  `torch.empty_like` + slice-assign. The tensor is 8 bytes
  (`batch_size = max_num_seqs = 2`) so the perf delta is invisible, but
  the idiom matches our P38 pattern and reads cleaner. P71 itself remains
  opt-in, default OFF.

- **`scripts/genesis_bench_v3.py`** — backported v4 fix: use
  `usage.completion_tokens` from final SSE chunk instead of counting raw
  delta chunks. Necessary because vLLM nightly batches stream deltas
  (3-5 tokens per chunk), so the old chunk-count was undercounting tokens
  by 3×, masking real throughput. Server-side bench was fixed already
  (validated v7.48 baseline at 165 tok/s vs old 51 tok/s misreading);
  public scripts now align.

### Notes

- Server-side bench tools (`/home/sander/Genesis_Project/vllm_engine/`)
  remain `genesis_bench_v3.py` (already patched) AND
  `genesis_bench_v4.py` (separate file kept for cross-check). Public repo
  ships only the corrected `genesis_bench_v3.py`.
- All `torch.cat` in `_genesis/` tree audited: only documentation comments
  remain referencing it as historical context.
- Snapshot tag for rollback: `pre-quick-wins-2026-04-27`.

---

## v7.51.1 — 2026-04-27 (Action #2/#3 evaluation + dev/public split)

Documentation-only update closing out the audit of two further candidates
from the vllm#40941 deep-dive (Action #2: OUTPUT_FP16 stage2 fold;
Action #3: `torch.cat` → slice-assign in continuation prefill).

### Action #2 — OUTPUT_FP16 stage2 fold: NOT APPLICABLE

PR #40941 adds `OUTPUT_FP16: tl.constexpr` to upstream
`vllm/v1/attention/ops/triton_decode_attention.py:_fwd_kernel_stage2`
to fold an fp32→fp16 cast into the `tl.store`. This kernel is used by
the upstream `_decode_attention` path (non-spec decode + ngram_gpu).

**Our P67 multi-query kernel does NOT use upstream `_fwd_kernel_stage2`** —
it is single-pass (no two-stage reduce-then-cast pattern), writes its
output directly inside the inner loop. So the OUTPUT_FP16 win is invisible
to our MTP K=3 verify path which is what dominates our production
workload.

The only path where this would help us is `start_no_spec_async.sh`
(no-spec mode using upstream decode kernel). For that path the win is
still small (one launch per token saved). Not worth a backport on the
MTP-default prod stack. Re-evaluate if/when we ship a no-spec variant
as a primary path.

### Action #3 — `torch.cat` → slice-assign: ALREADY DEPLOYED via P38

PR #40941 replaces `k_full = torch.cat([k_cached_trim, key_chunk])` with
pre-allocate-then-slice in upstream `_continuation_prefill`. This is
**already what our P38 (`patch_38_tq_continuation_memory.py`) does** when
its prealloc pool is wired:

```python
k_full[:cached_len].copy_(src)           # cached portion
k_full[cached_len:seq_len].copy_(key_chunk)  # new chunk
```

The fallback path (when prealloc not wired — e.g. AMD/CPU tests) still
uses `torch.cat`, which is the upstream pre-#40941 behaviour. That's
fine — fallback is rare and correctness-preserving by design.

The remaining `torch.cat` sites in our `_genesis/` tree are:
- `block_verify_sampler.py:131,275` — P71, opt-in default OFF, builds
  a 2-element `cu_start` tensor (literally 8 bytes). Marginal.
- `dequant_buffer.py` — only in comments documenting the pattern P38
  replaces.

**Net: nothing left to extract from PR #40941 that we're not already
doing.** v7.48 P38/P40 shared-pool work covered this ground.

### Repo housekeeping (separate from Action items)

- 22 dev kernel artifacts (`p67_dev/`), 2 backup tarballs (`p67_backups/`),
  and `docs/DISCUSSION_DRAFT_NOONGHUNNA.md` moved to private repo
  `Sandermage/p67-genesis-kernel`. Public patcher repo now ships only
  production-ready patches + supporting infra.
- Root-level Python harness/bench files moved under `scripts/` for tidier
  layout: `genesis_bench_v3.py`, `genesis_quality_harness.py`,
  `genesis_context_sweep.py`, `genesis_longbench_runner.py`.
- `.gitignore` hardened to prevent re-adding dev artifacts.
- Server-side backup `/home/sander/genesis-backups/v7.50-stable-20260427_0202/`
  contains full restore set (tar of `_genesis`, scripts, compile cache,
  bench tools + RESTORE.md).

### Snapshot tags (rollback-safe)

- `v7.50-stable-2026-04-27` — pre-Step-D state
- `v7.51-stable-2026-04-27` — current production (P67 exp2+FLT_MAX)
- `pre-step-d-2026-04-27` — transient, before Step D sweep
- `pre-action-2-2026-04-27` — transient, before Action #2 audit (no code changes resulted)

### Next sprints (deferred)

- **Tier 3 H** — re-fuse split-M with per-row online softmax. Multi-day
  refactor, needs FP64 reference gate (numerical correctness regression
  suite). Expected gain: +8-15%. Risk: medium (numerical drift across
  ~256 KV iterations is what split-M originally fixed).
- **Tier 3 I** — 2D split with `temp_size=32` (vllm#38786 backport).
  Multi-day, needs context-window sweep (4K → 256K). Expected gain:
  +8-15% specifically on long context.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.51 — 2026-04-27 (P67 softmax: tl.exp2 + -FLT_MAX sentinel; Step D rejected)

Two corrections to `p67_multi_query_kernel.py` softmax inner loop, derived
from deep-dive of upstream PR vllm#40929 (DeepSeek-V4 Triton fallback
kernels) — they apply textbook FlashAttention-2 numerical idioms that
our v7.34 split-M kernel had missed.

### Changed (`vllm/_genesis/kernels/p67_multi_query_kernel.py`)

- **`tl.exp` → `tl.exp2`** for online-softmax `α_t` and `P_t` updates.
  Triton's `tl.exp2` maps directly to the hardware `ex2.approx.f32`
  PTX instruction; `tl.exp` is synthesized as `ex2(x * log2e)` so adds
  one extra fp multiply per softmax step. Pre-multiplying by
  `LOG2E = 1.4426950408889634` once is the standard FA2 idiom.
- **`float("-inf")` → `-3.4028234663852886e38`** (`-FLT_MAX`) for masked-out
  attention scores. `inf*0 = NaN` in fp32 accumulator can poison the
  online-softmax across subsequent KV iterations; FLT_MAX gives the same
  effective masking via `tl.exp2(very_negative)` clamping to 0, but
  without NaN risk.

### Empirical (validated 2026-04-27, Qwen3.6-A3B-FP8 + MTP K=3)

| Metric | v7.50 | v7.51 | Δ |
|---|---|---|---|
| **Stability mean (10 runs)** | 157.6 | **167.2** | **+6.1%** |
| @ 1024 tok | 132.0 | **146.5** | **+11.0%** |
| @ 2048 tok | 137.7 | 142.0 | +3.1% |
| @ 128 tok | 172.5 | 169.4 | -1.8% |
| @ 256 tok | 160.1 | 150.8 | -5.8% (3-run high CV) |
| @ 512 tok | 144.9 | 135.9 | -6.2% (3-run high CV) |
| Quality 30-shot | 30/31 PASS | **30/31 PASS** | preserved |
| Tool-call | 2/2 PASS | 2/2 PASS | preserved |

Stability mean (10-run, low CV) is the load-bearing number. Mid-length
3-run speed tests have CV up to 11% — within noise. Long-generation
(1024+) shows clear consistent improvement.

### Step D (@triton.autotune) — REJECTED

Manual sweep of 5 alternative configs vs production
(BLOCK_KV=32, NUM_WARPS=8, num_stages=3):

| BLOCK_KV | NUM_WARPS | tok/s | vs baseline 157 |
|---|---|---|---|
| 16 | 4 | 151.0 | -3.8% |
| 16 | 8 | 148.9 | -5.2% |
| 32 | 4 | 149.1 | -5.0% |
| 64 | 4 | 147.9 | -5.8% |
| 64 | 8 | 149.6 | -4.7% |

All alternatives regressed. Current `(32, 8, stages=3)` IS the optimum
for our Ampere SM 8.6 + dequant-heavy workload. No autotune needed —
the search space has a clean global maximum at the current setting.
Rejection rationale recorded in P67 docstring with each config row.

### Notes

- `LOG2E = 1.4426950408889634` is precomputed at compile time inside the
  inner loop body — Triton constant-folds it.
- `_FLT_MAX_NEG = -3.4028234663852886e38` is the IEEE 754 most-negative
  finite fp32 value. It survives all subtraction operations within the
  online-softmax range without underflow.
- Action #1 derived from research-agent deep-dive of vllm#40929 DeepSeek-V4
  Triton fallback kernels (PR doesn't apply to us as a model, but the
  softmax idioms inside transfer cleanly to our k8v4 verify path).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.50 — 2026-04-27 (Tier 1 Step C: P67 cache_modifier + tl.range hints)

Backport of [vllm#33529](https://github.com/vllm-project/vllm/pull/33529)
("Triton MLA perf fixes", merged 2026-04-02) Triton compiler hints into our
P67 multi-query attention kernel. Memory-traffic optimizations only — zero
arithmetic change.

### Changed (`vllm/_genesis/kernels/p67_multi_query_kernel.py`)

- **`tl.range()` instead of plain `range()`** for the outer KV loop —
  explicit Triton pipelining hint. Lets the compiler overlap `cp.async`
  loads with prior-iteration MMA on Ampere.
- **`cache_modifier=".cg"`** on K/V dequant raw loads (`KV_cache_ptr +
  k_addrs / val_addrs`) — streaming reads that should NOT pollute L1.
  L2-direct frees L1 capacity for Q + scales.
- **`cache_modifier=".ca"`** on Q load, `Block_table_ptr` lookup, and
  scale/zero loads (`sc_lo`, `sc_hi`, `zr_lo`, `zr_hi`) — these are
  reused inside the CTA across all KV iterations. Pinning them in L1
  saves repeated DRAM round-trips.
- **Hoisted `kv_head * stride_cache_head`** out of the inner KV loop
  (`_kv_head_byte_offset` precomputed once per CTA) — invariant across
  all per-tile `slot_bases` calculations. Triton -O2 would also hoist
  this but explicit form matches upstream MLA decode style.

### Empirical (validated 2026-04-27, 2× RTX A5000 + Qwen3.6-A3B-FP8 + MTP K=3)

| max_tokens | v7.48 | v7.50 | Δ |
|---|---|---|---|
| 64 | 188.6 | 191.0 | +1% |
| 256 | 145.6 | **160.1** | **+10%** |
| 512 | 141.8 | 144.9 | +2% |
| 1024 | 132.8 | 132.0 | ~0% |
| 2048 | 129.2 | 137.7 | +6.5% |

Stability mean 157-162 tok/s (within v7.48 noise band). Quality 30/31
PASS unchanged. Tool-call regression 2/2 unchanged. Long-context probe
16K-160K all PASS at GMU 0.90.

### Notes

- All hints are memory-traffic only — `cache_modifier` is a PTX-level
  cache-policy attribute, not arithmetic. Numerical correctness verified
  via quality harness (no token-level deviations from v7.48 baseline).
- `tl.range()` enables Triton 3.x async-copy pipelining (`num_stages>1`
  in our autoconfig). On Triton 2.x it falls back to a plain loop —
  graceful degrade.
- Tested ONLY on `cache_modifier=".cg"`/`".ca"` literals supported by
  Triton 3.x on Ampere (sm_86). On older Triton, the modifiers are
  ignored — kernel still correct, just no cache-policy hint applied.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.49 — 2026-04-27 (P79d retired + P79c improved per upstream review)

Two small but important corrections to the v7.46 async-safety patch trio,
based on upstream maintainer feedback received within 24h of v7.48 push.
Tested versions unchanged from v7.48 (vLLM `dev212+g8cd174fa3`, PyTorch
`2.11.0+cu130`, Triton `3.6.0`, CUDA 13.0, driver 580.126.09, 2× RTX A5000).

### Removed

- **P79d retired completely** (`vllm/_genesis/wiring/patch_79d_preempt_async_discard.py` + dispatcher entry + apply_all register).
  - Reason: njhill (vLLM core maintainer) explicitly confirmed in
    [vllm#38624](https://github.com/vllm-project/vllm/pull/38624) that the
    asymmetry P79d "fixed" is **intentional**: the regular `_preempt_request`
    removes the request from the next step entirely, so placeholder state is
    never re-read; only `reset_prefix_cache` re-admits it. The backport
    targeted a non-bug.
  - CodersAcademy006 (original PR author) acknowledged the static-analysis
    miss and committed to closing #38624 with a clarifying comment.
  - Genesis prod is unaffected (P79d was opt-in, default-off, never enabled
    in our `start_mtp.sh`). Removal is preventive — keeps the patcher
    surface clean and avoids any operator accidentally enabling a
    misguided modification.

### Improved

- **P79c smarter cleanup** (`vllm/_genesis/wiring/patch_79c_stale_spec_token_cleanup.py`):
  - Old behaviour: cleared **any** `spec_token_ids` for unscheduled running
    requests — risked wiping **real draft token IDs** (positive ints from
    MTP / EAGLE / ngram), not just `-1` placeholders. Could corrupt MTP
    state across budget-exhaustion cycles.
  - New behaviour (matches the spirit of the emerging canonical fix
    [vllm#40768](https://github.com/vllm-project/vllm/pull/40768) by jvlunteren):
    1. Only clear when `spec_token_ids` is **all `-1`** (`all(t == -1 for t in ids)`).
       Real draft tokens preserved.
    2. **`prev_step_scheduled_req_ids` membership gate** — if request was in
       the previous worker step, placeholders may still be consumed by async
       input prep; we leave them alone. Otherwise (new request not in prev
       step) → safe to clear.
  - Drift detector unchanged — when #40768 (or the eventual canonical fix)
    merges and adds `_consume_spec_decode_tokens_for_step`, our P79c
    self-skips and the upstream takes over.
  - Still opt-in via `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1`.
    Genesis prod (sync ngram, max_num_seqs=2) still doesn't engage it —
    only protects high-concurrency multimodal users on async + EAGLE/MTP.

### Tracker delta (since v7.48 push)

- **vllm#38624 (P79d source)**: dead per maintainer — already retired here.
- **vllm#40610 (P79b source)**: still draft, no human review — backport stays.
- **vllm#37629 (P79c source)**: active discussion (benchislett asked for
  non-multimodal repro; haosdent committed to providing one). Watch for v2
  with proper root-cause fix.
- **vllm#40925 (P81 source)**: open, mergeable, blocked on first-time
  contributor label gate. Backport stays. Will retire when merged.
- **vllm#40768 (canonical fix for the bug class P79c addresses)**: NEW PR,
  introduces `_consume_spec_decode_tokens_for_step` + dedicated
  `num_pending_async_spec_placeholders` Request field. Direct supersession
  candidate for our P79c — when it lands we drop the patch entirely.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.48 — 2026-04-27 (memory shared-pool sprint + P81 backport + driver 580)

Tested on **vLLM 0.19.2rc1.dev212+g8cd174fa3** (nightly image
`vllm/vllm-openai:nightly` ID `10c7a6ba51c6`, PyTorch 2.11.0+cu130,
Triton 3.6.0) on **NVIDIA driver 580.126.09 + CUDA 13.0**, 2× RTX
A5000 (Ampere SM 8.6), Qwen3.6-35B-A3B-FP8 with TurboQuant k8v4 KV
cache and MTP K=3 spec decode.

### Added

- **P81 — fp8 block-scaled MM low-M decode tuning**
  (`wiring/patch_81_fp8_block_scaled_m_le_8.py`):
  - Backport of [vllm#40925](https://github.com/vllm-project/vllm/pull/40925)
    (tonyliu312, OPEN as of 2026-04-26)
  - Opt-in via `GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1`
  - Specializes default `w8a8_triton_block_scaled_mm` config for M ≤ 8
    (single-request decode + MTP K=3 verify): `BLOCK_SIZE_M` 64 → 16,
    `num_stages` 2 → 3 (non-ROCm)
  - Direct hit for our prod (Qwen3.6-A3B FP8 + max_num_seqs=2,
    no pre-tuned JSON for A5000)
  - Empirical (per upstream PR on GB10 sm_121): +23% median decode
  - Drift detector: presence of `if M <= 8:` literal without Genesis
    marker → upstream PR merged → auto-skip

- **`vllm/_genesis/buffer_mode.py`** — centralized buffer-mode toggle:
  - Reads `GENESIS_BUFFER_MODE=shared|per_layer` env (default `shared`)
  - Per-patch override via `GENESIS_BUFFER_MODE_<PID>` (e.g. P38, P40)
  - `shared` = singleton pool via `GenesisPreallocBuffer` (memory-efficient,
    saves multi-GB on long-context)
  - `per_layer` = legacy attached-attribute path (rollback safety)

### Memory-opt sprint

Driver 570 → 580 upgrade brought CUDA 13.0 PyTorch which adds ~3 GB
allocator overhead. To restore long-context capability while staying
at GMU 0.90+, audited prealloc patches:

- **8 of 9 patches** (P22/P26/P28/P36/P37/P39/P44/P46) **already use
  shared singleton** via `TurboQuantBufferManager` /
  `GenesisPreallocBuffer` / `gdn_core_attn_manager` /
  `FlaKktBufferManager`. The `setattr(layer, ...)` only attaches 36
  references to a single registered buffer; per-layer attribute lookup
  is for fast-path access, not duplicated allocation.

- **P38 was the real waste** — `_genesis_continuation_prefill` had a
  fresh `torch.empty(buf_shape, ...)` fallback when `seq_len` exceeded
  current buffer, allocating per-call growth. Fixed
  `wiring/patch_38_tq_continuation_memory.py` to use `buffer_mode_for`:
  `shared` mode now allocates **one max-size buffer** per
  (Hk, D, dtype, device) signature via `GenesisPreallocBuffer`, slice
  to actual `alloc_len` per call. Single namespace, no growth churn,
  no per-layer waste.

- **P40 fallback** (TQ grouped decode `mid_o`/`output`/`lse` per-call
  `torch.empty` when `buf_holder` not pre-attached) similarly fixed via
  `buffer_mode_for("P40")` shared singleton with max-shape (max_B,
  Hq, NUM_KV_SPLITS, D+1) registered through `GenesisPreallocBuffer`.

### Empirical (validated v7.48 baseline)

| Metric | v7.48 (driver 580 + P81 + shared P38/P40) | vs v7.13 (driver 570 + per-layer) |
|---|---|---|
| Throughput mean | **160-190 tok/s** | 130-143 tok/s = **+15-30%** |
| Quality 30-shot harness | 30/31 PASS (96.8%) | 30/31 PASS |
| Tool-call regression | 2/2 PASS | 2/2 PASS |
| Long-ctx 16K-160K needle | ALL PASS | 16K-128K PASS |
| Long-ctx 200K | PASS (153K server tokens) | OOM |
| **GMU at which 200K runs** | **0.90** (Sander obligatory range MET) | 0.91 limit |
| Production launch script | `scripts/launch/start_mtp.sh` (updated) | unchanged |

### Notes

- Env-driven `GENESIS_BUFFER_MODE=shared` is the new default. Set to
  `per_layer` if shared pool ever shows regression on a different
  model/config — purely a rollback knob, not expected for normal use.
- The shared pool requires sequential layer execution within the
  forward pass (which is true for TP + non-PP + sync-scheduling — our
  config). Anyone adding pipeline parallelism or multi-stream pipelined
  layers should re-evaluate.
- Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.46 — 2026-04-26 (async × spec-decode safety patches — opt-in)

Three additive backports of OPEN upstream PRs that fix async-scheduling
race conditions on EAGLE/MTP/ngram_gpu paths. **All three default-off:**
Genesis prod (sync ngram, max_num_seqs=2) gains nothing direct — these
protect users who run `--async-scheduling` + spec-decode.

### Added

- **P79b — Async × spec-decode proposer-sync** (`wiring/patch_79b_async_proposer_sync.py`):
  - Backport of [vllm#40610](https://github.com/vllm-project/vllm/pull/40610) (OPEN draft)
  - Opt-in via `GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=1`
  - Wraps `GPUModelRunner.sample_tokens()` to re-record `prepare_inputs_event`
    in `finally:` AFTER spec-decode proposer GPU work completes
  - Fixes happens-before race where next batch's `_update_states` could
    mutate persistent block_table while previous batch's proposer was
    still reading on GPU — symptom: nondeterministic stale state on
    async + EAGLE/MTP/ngram_gpu
  - Drift detector: presence of `_sample_tokens_impl` symbol without
    Genesis marker → upstream merged → auto-skip
  - Verified on dev205+g07351e088: applies cleanly, file compiles,
    method reload picks up new structure, idempotent

- **P79c — Stale spec_token_ids cleanup** (`wiring/patch_79c_stale_spec_token_cleanup.py`):
  - Backport of [vllm#37629](https://github.com/vllm-project/vllm/pull/37629) (OPEN, fixes #36906)
  - Opt-in via `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1`
  - Adds cleanup pass after main scheduling loop in `Scheduler.schedule()`
    that clears `spec_token_ids` for any running request not in
    `num_scheduled_tokens`
  - Fixes EAGLE3 + async high-concurrency CUDA device-side assert
    in `F.embedding()` from stale `-1` placeholder leak
  - Triggered when token budget exhausts before scheduler visits all
    running requests; not multimodal-specific (PR's regression test
    proves text-only with sufficient concurrency reproduces)
  - Verified on dev205+g07351e088: applies cleanly, file compiles,
    idempotent

- **P79d — Preempt async-discard** (`wiring/patch_79d_preempt_async_discard.py`):
  - Backport of [vllm#38624](https://github.com/vllm-project/vllm/pull/38624) (OPEN, CodersAcademy006)
  - Opt-in via `GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD=1`
  - Adds 2 lines to `_preempt_request()` that set
    `num_output_placeholders=0` + `discard_latest_async_tokens=True`
  - Currently set ONLY in `reset_prefix_cache()` — scheduler-loop
    preemption path bypasses cleanup, leading to duplicated tokens
    after request resume on async paths ("the the", "of of")
  - SAFER than upstream PR: additive only — does NOT remove the
    existing block from `reset_prefix_cache()` (defensive)
  - Drift detector: counts `discard_latest_async_tokens = True`
    occurrences; ≥2 → upstream merged → auto-skip
  - Verified on dev205+g07351e088: applies cleanly, count goes 1→2,
    marker present, idempotent on re-apply

### Investigation notes (no patch)

- `docs/DRAFT_38903_async_pp_contamination.md` — local draft on
  [vllm#38903](https://github.com/vllm-project/vllm/issues/38903)
  (cross-request data contamination on PP>1 + async + multi-node).
  Severe privacy bug but Genesis cannot reproduce (TP=2, PP=1,
  single-node, single-user). Document includes proposed config-level
  bandaid, recommendation NOT to ship it (over-broad, no reproducer),
  and pointer to research-agent's Section 5 Cat-5v embedding-input
  invariant guard as a multi-bug defensive layer worth implementing.

### Empirical findings (no patch — data only)

- **P80 ngram_gpu+async verification on dev205+ pin**: 3-shot bench on
  Qwen3.6-A3B FP8 + `--async-scheduling` + ngram_gpu method (testing
  whether [vllm#37150](https://github.com/vllm-project/vllm/issues/37150)
  fix shipped). Result: 35-43 tok/s mean ≈40 — works without error
  (no cascade, normal output) but SLOWER than sync ngram CPU (46 tok/s).
  Same pattern as our MTP+async finding: async overhead > savings on
  single-user max_num_seqs=2 setups.
- Confirms #37150 fix IS active in our pin (no 1.22% acceptance
  pathology), but ngram_gpu+async at single-user is net-negative —
  use sync ngram CPU instead for our workload class.

### Upstream-watch deltas

These three PRs are independent of the TurboQuant workspace cluster
tracked above. Each has its own drift marker; none conflict with
P22/P26/P67/P67b. After any of #40610/#37629/#38624 merges, the
respective drift detector auto-skips the corresponding Genesis patch.

---

## Upstream-watch — pending rebase work (added 2026-04-26)

Three competing upstream PRs target the TurboQuant decode scratch workspace
that our P22 / P26 / P67b stack already addresses. We monitor + auto-skip
when any merges. Action plan per PR:

| Upstream PR | Author | Drift marker | Genesis impact |
|---|---|---|---|
| **#40798** (likely winner) | Bot1822 | `_reserve_turboquant_decode_workspace` symbol in `vllm/v1/worker/gpu_model_runner.py` | (1) P22 auto-skips via drift detector. (2) **P67b needs rebase** — the PR REMOVES `buf_holder` kwarg from `triton_turboquant_decode_attention`. Drop these 4 explicit args from `patch_67b_spec_verify_routing.py:131-156`: `mid_o_buf`, `output_buf`, `lse_buf`, `buf_holder=layer`. Routing logic itself unchanged. |
| #40706 (backup) | lesj0610 | `reserve_turboquant_decode_workspace` symbol in `vllm/v1/attention/backends/turboquant_attn.py` | (1) P22 auto-skips. (2) P67b unchanged — preserves `buf_holder` fallback. |
| #40655 | bhoomit | `_init_turboquant_buffers` REMOVED from `TurboQuantAttentionImpl` | (1) P22 auto-skips. (2) P67b unchanged. CHANGES_REQUESTED upstream — less likely to land. |

**Drift detector**: `wiring/patch_22_tq_prealloc.py:_check_upstream_tq_workspace_drift()` probes for all 3 markers. When any matches, P22 returns `("skipped", "PR #XXXXX merged ...")` — ready for next sync without manual intervention.

**P26 (prefill output)**: orthogonal to all 3 PRs (they target decode path, P26 covers prefill). KEEP.

**Our PR #40914**: complementary, not competing. Routing fix vs workspace dedup are separate axes.

---

## v7.11.0 — 2026-04-25 (spec-decode workaround + diagnostic tooling)

**Investigation + opt-in workaround for [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831)** — TurboQuant × any speculative decoding (MTP or ngram) produces degenerate token loops on structured outputs.

### Added

- **P56 — TQ spec-decode safe-path guard** (`wiring/patch_56_spec_decode_decode_path_guard.py`):
  - Opt-in via `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1` (off by default)
  - 5-line text-patch on `turboquant_attn.py` tightens `_prefill_attention` continuation fast-path entry from `q_len ≤ _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`
  - Spec-decode batches (q_len > 1) now route through `_continuation_prefill`'s `flash_attn_varlen_func(causal=True)` — causal-correct
  - Closes Layer 1 (catastrophic XML/JSON loops); Layer 2 (token duplication, e.g. `for for`, `age age`, `parameter parameter`) remains and is upstream's territory
  - Registered in `apply_all.py` between P26 and P44

- `scripts/sequential_backend_probe.py` — 9-prompt diagnostic probe set covering smoke / narrative / tool calls (no-thinking + thinking) / JSON / needle short+medium / code / structured XML. `run` subcommand fires the set against any vLLM endpoint and writes JSONL; `diff` subcommand compares two such logs side-by-side with degenerate-pattern detection.

- `scripts/dual_backend_diagnostic_proxy.py` — FastAPI proxy on :9000 forwarding each request to two backends concurrently. Captures both responses byte-for-byte, computes structural diff, detects degenerate patterns. Useful when concurrent backends fit (we currently fall back to sequential probing because TP=2 saturates our 2× A5000).

### Verified on Genesis pin `fe9c3d6c5`

- 2× RTX A5000 (Ampere SM 8.6), TP=2
- Qwen3-Next-35B-A3B-FP8 (MoE hybrid), `kv_cache_dtype=turboquant_k8v4`
- ngram spec-decode `n=3` (chosen so result doesn't depend on MTP draft head)
- Reproduced #40831 catastrophically without P56 (`tool_calls=[]`, content=`<parameter=parameter=unit>...</parameter>×16+`)
- With P56: `tool_calls` populated, narrative coherent, no infinite loops
- Layer 2 token-duplication probed via 9-prompt diff against prod baseline; documented in upstream comment

### Upstream interactions (this release)

- [#40807 issuecomment-4316663581](https://github.com/vllm-project/vllm/issues/40807#issuecomment-4316663581) — pointed at P44+P23 as fix direction for the CUDA graph crash (noonghunna's first bug)
- [#40124 issuecomment-4316828133](https://github.com/vllm-project/vllm/issues/40124#issuecomment-4316828133) — replied to noonghunna's heads-up; promised the test we then ran
- [#40831 issuecomment-4317214311](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) — full Layer 1 root cause + P56 workaround + Layer 2 finding

### Files touched

- `vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py` (new, ~165 lines)
- `vllm/_genesis/patches/apply_all.py` (+46 lines for P56 registration)
- `scripts/sequential_backend_probe.py` (new, ~225 lines)
- `scripts/dual_backend_diagnostic_proxy.py` (new, ~265 lines)
- `README.md` — v7.11 What's-new section, P56 in opt-in roster, upstream tracking with issuecomment IDs, scripts/ in architecture
- `vllm/_genesis/CHANGELOG.md` — this entry

---

## v7.9.0 — 2026-04-24 (runtime architecture-dispatch detection)

**Defense-in-depth layer 2: detect which patches need to fire before work begins.**

### Added

- `model_detect.py` — cached `get_model_profile()` returns `(moe, hybrid, turboquant)`
  - `is_moe_model()` — Qwen3-MoE / Mixtral / DeepSeek / Gemma-4-MoE / architecture + model_type heuristics
  - `is_hybrid_model()` — Qwen3-Next `layer_types`, Mamba, GDN, SSM detection
  - `is_turboquant_active()` — config-level `kv_cache_dtype` check (layer-level is P51 in `dequant_buffer.py`)
  - `log_skip(patch, reason)` — uniform single-line dispatch log format
  - `clear_for_tests()` — cache reset for unit tests
  - Conservative fallback: unknown config → True for all flags (patches still apply, their own guards decide)

- **P51 — TQ-active runtime detection** in `kernels/dequant_buffer.py::ensure_turboquant_buffers`
  - Reads `impl.kv_cache_dtype`; early-returns with single log if non-TurboQuant
  - Saves ~516 MiB / rank on FP16-KV + `auto` deployments where TQ text-patches graceful-skip but preallocs would fire
  - `_p51_logged` flag avoids log spam across all model layers (one log per impl)

- **P52 — MoE-active dispatch gate** wired into `wiring/patch_{24,31,37}_*.py`
  - Skips P24 (MoE num_warps overlay), P31 (grouped-topk fp32 upcast), P37 (intermediate-cache pool) on dense models
  - Single log line per skipped patch at apply time; no runtime overhead thereafter

- **P53 — Hybrid-active dispatch gate** wired into `wiring/patch_{28,34,39,46}_*.py`
  - Skips P28 (GDN core-attn rewire), P34 (Mamba zero-collapse guard), P39a (FLA kkt pool), P46 (GDN gating pool) on pure-attention models
  - All targets still graceful-skip without P53 (their text-patch anchors wouldn't match), but the dispatch log now explains *why*

- `tests/test_model_detect.py` — 19 tests covering MoE detection across architectures, hybrid detection, TQ detection, conservative fallback, caching, log helper
- `tests/test_p51_tq_active.py` — 8 tests covering fp8/auto/fp16 skip, single-log-per-impl, legacy-impl backward compat, TQ-active passthrough

### Changed

- `kernels/dequant_buffer.py::ensure_turboquant_buffers` now early-returns on non-TQ impls before any config resolution work
- Wiring apply() docstrings updated to reference P52/P53 gates where applicable
- Root `README.md` rewritten for v7.9 with compatibility matrix, installation guide, patch roster, upstream tracking

### Upstream correspondence

Re-audit of `vllm-project/vllm` since 2026-04-24 surfaced:
- **#40807** (OPEN) — TurboQuant + spec-decode capture crash; reporter namechecks Sander's Patch 23. Our P44 aligns.
- **#40792** (OPEN) — TQ k8v4 GQA head grouping; may supersede our P40. Diff + bench pending.
- **#40798** (OPEN) — TQ scratch workspace across layers; superset of #40655+#40706. May conflict with P28 anchor.
- **#40794** (MERGED 2026-04-24) — MoE unpad routed output; smoke test on Qwen3.6-35B-A3B pending.
- **#40420** (OPEN) — TQ continuation-prefill OOM at 185k; adding ≥150k regression to integration gate.

No PR posted upstream without explicit user approval (per `feedback_no_push_without_explicit_approval`).

---

## v7.8.5 — 2026-04-24 (cross-quantization validation)

Validated v7.8 on three configurations: FP8 prod / AWQ 4-bit / FP16-KV 32k.

**Results**: 28 applied / 0 failed across all three. 3× 256k stable on FP8 + AWQ. AWQ frees ~9 GiB/rank → 2.5× KV capacity (1.099M → 2.787M tokens). Speed: AWQ 1-4% slower than FP8 (4-bit dequant cost on SM 8.6). Linear degradation unchanged: `1/tgs ≈ 0.007 + 2.4e-5 × ctx`.

**Finding**: TQ preallocated buffers waste ~516 MiB/rank on FP16-KV deployments where TQ is inactive — led to P51 in v7.9.

## v7.8.0 — 2026-04-24 (interface guards + middleware)

### Added

- **P49 — interface contract validation** (`interface_guard.py`, ~240 lines)
  - `GenesisInterfaceMismatch` exception
  - `validate_impl(impl, required_attrs, required_methods, optional_attrs, role)` helper
  - `validate_method_signature(method, expected_params)` — catches renamed params
  - `assert_shape_compat(t, expected, msg)` — runtime shape drift detection
  - `describe_impl(impl)` — diagnostic snapshot
  - `ANY` sentinel — presence-only check (used for Triton `@triton.jit` kernels that aren't `callable()` in Python sense)
  - Wired into P22, P38, P39a as pre-flight guards (defense layer 1)

- **P50 — ASGI `ResponseCacheMiddleware`** (`middleware/response_cache_middleware.py`, ~280 lines)
  - Drop-in ASGI middleware for any FastAPI/Starlette app (target: cliproxyapi:8330)
  - Deterministic cache key (JSON `sort_keys=True`)
  - `stream=True` + sampled requests (`temp>0`, `top_p<1`, `top_k>1`) NOT cached by default
  - Graceful degradation on cache errors (silent miss)
  - `x-genesis-cache: HIT|MISS` header for diagnostics

- 18 tests in `test_interface_guard.py` (validate, sig, shape, describe)
- 25 tests in `test_response_cache_middleware.py` (key extraction, ASGI flow, error handling)

### Fixed

- P39a initial false-positive: Triton `@triton.jit` `chunk_scaled_dot_kkt_fwd_kernel` isn't Python-callable. Switched to `required_attrs={...: ANY}` (presence check) instead of `required_methods` (callable check). The guard correctly caught the edge case — API usage corrected.

### Tests

Full unit suite: 605 passed / 8 skipped / 0 failed.

---

## v7.0.0-dev — 2026-04-24

**Major architectural shift**: migrate from monolithic text-replacement overlay (`patch_genesis_unified.py`, ~3000 LOC) to modular professional package.

### Added

- `vllm/_genesis/` package structure (upstream-compatible namespace)
- `guards.py` — canonical vendor/chip/model/dependency detection
  - Vendor identity: `is_nvidia_cuda()`, `is_amd_rocm()`, `is_intel_xpu()`, `is_cpu_only()`
  - NVIDIA compute capability: `get_compute_capability()`, `is_sm_at_least(major, minor)`, arch predicates (`is_ampere_consumer()`, `is_hopper()`, `is_blackwell()`, etc.)
  - AMD architecture: `is_rocm_cdna2()`, `is_rocm_cdna3()`, `is_rocm_rdna()` via `_GCN_ARCH` parsing
  - Dependency versions: `get_torch_version()`, `get_transformers_version()`, `get_vllm_version_tuple()`, `is_transformers_v5_plus()`, `is_torch_211_plus()`
  - Model architecture: `is_model_arch(cfg, arch_name)`, family helpers (`is_qwen3_family`, `is_deepseek_v3`, etc.)
  - Backend detection: `has_turboquant_support()`, `is_marlin_selected()`, `is_flash_attn_backend()`
  - Path resolution: `vllm_install_root()`, `resolve_vllm_file()` — replaces hardcoded `/usr/local/lib/python3.12/` paths (works on any Python version, Mac/Linux/Docker slim)
  - Diagnostic: `platform_summary()` returns full JSON-serializable platform info

- `prealloc.py` — `GenesisPreallocBuffer` framework
  - Class-level registry for shared tensor allocation
  - `get_or_create(namespace, shape, dtype, device, zero_init)` — fresh or cached
  - `slice_to(buf, n, dim)` — pointer-stable view (CUDA graph safe)
  - `get_registry_info()` — diagnostic JSON of all allocations
  - `clear_for_tests()` — test helper (warns if called outside pytest)

- `kernels/router_softmax.py` — **Patch 31** implemented
  - Drop-in replacement for `torch.softmax` in MoE routers
  - Fp32-upcast intermediate prevents bf16 mantissa collision
  - Fixes non-deterministic top-k routing on Qwen3-MoE (pre-SM90)
  - `router_softmax()` and `router_softmax_preserving_mask()` variants
  - Platform-universal: CUDA / ROCm / XPU / CPU all supported

- `kernels/dequant_buffer.py` — **Patch 22** skeleton (Phase 2 target)
  - `TurboQuantBufferManager` class with platform guard
  - Designed for profiler-visible KV buffer pre-allocation

- `kernels/gdn_dual_stream.py` — **Patch 7** skeleton (Phase 2 target)
  - `DualStreamDispatcher` with platform-aware fallback
  - NVIDIA parallel, ROCm HIP attempt, XPU/CPU sequential

- `kernels/marlin_tuning.py` — **Patch 17/18** skeleton (Phase 2 target)
  - Per-SM optimal `block_size_m` auto-selection
  - Env overrides: `VLLM_MARLIN_MOE_BLOCK_SIZE_M`, `_NUM_WARPS`, `_NUM_STAGES`

- `kernels/fp8_dispatcher.py` — **Patch 1/2** skeleton (Phase 2 target)
  - `requires_marlin_fp8_fallback()` — SM<8.9 detection
  - Per-arch routing logic

- `patches/apply_all.py` — new orchestrator replacing monolithic patcher
  - Decorator-based patch registration (`@register_patch("P31 ...")`)
  - `PatchStats` with counts and per-patch details
  - CLI entrypoint: `python3 -m vllm._genesis.patches.apply_all`
  - Exit codes: 0 success / 1 patch failure / 2 setup error
  - Stub registration for Patch 31 (full implementation Phase 2)

- `patches/upstream_compat.py` — upstream PR marker registry
  - Central tracking of all upstream fixes Genesis mirrors
  - Used by Layer 3 (upstream merge) defensive checks
  - Coverage: #39016, #39391, #39953, #40060, #40105, #40159, #40172, #40194, #40384, #40572, #40633, #38479

- `tests/conftest.py` — pytest fixtures
  - `cuda_available`, `rocm_available`, `nvidia_cuda_available` platform fixtures
  - `reset_genesis_prealloc` — clear registry before/after test
  - `deterministic_seed` — torch.manual_seed(42)
  - Custom markers: `cuda_required`, `rocm_required`, `gpu_required`, `slow`
  - Auto-skip GPU tests on CPU-only hosts

- `tests/test_guards.py` — comprehensive guards test coverage
  - TestVendorIdentity (6 tests)
  - TestComputeCapability (5 tests)
  - TestDependencyVersions (4 tests)
  - TestModelArchDetection (4 tests)
  - TestBackendDetection (2 tests)
  - TestPathResolution (3 tests)
  - TestPlatformSummary (2 tests)

- `tests/test_prealloc.py` — `GenesisPreallocBuffer` test coverage
  - TestGetOrCreate (7 tests)
  - TestSliceTo (6 tests)
  - TestRegistryInfo (3 tests)
  - TestPointerStability (2 tests) — CRITICAL for CUDA graph
  - TestClearForTests (2 tests)
  - TestCUDABehavior (2 tests)

- `tests/test_router_softmax.py` — Patch 31 TDD test suite
  - TestRouterSoftmaxDeterminism (3 tests)
  - TestRouterSoftmaxDtypePreservation (5 tests, parametrized)
  - TestRouterSoftmaxMathematicalCorrectness (5 tests)
  - TestRouterSoftmaxPlatformSafety (4 tests)
  - TestRouterSoftmaxEdgeCases (3 tests)
  - TestRouterSoftmaxPerformanceCUDA (1 test, CUDA-gated)

- `README.md` — package documentation with usage, testing, migration status
- `CHANGELOG.md` — this file

### Design decisions (why this structure)

1. **Why `vllm/_genesis/` namespace**: placed inside vllm's package layout so installation via overlay mount works without PYTHONPATH manipulation. Leading underscore marks it as "private" (Genesis-specific, not upstream API).

2. **Why separate `kernels/` and `patches/`**: clean separation between WHAT the code does (kernels) and HOW it integrates (patches). When we submit upstream PRs, we submit kernels/ directly — patches/ is just the bridging overlay.

3. **Why TDD discipline**: project rule. "Test-first for new functionality." Also mandatory for Patch 28 (GDN prealloc) to prevent repeating Patch 19's revert (−30% throughput, 188× stdev).

4. **Why `@functools.cache` on guards**: NVML probe and vllm.platforms queries are ~1ms. Cached after first call (~50ns). At 20+ patches × startup = 20ms vs 1μs difference.

5. **Why `vllm_install_root()` helper**: replaces hardcoded `/usr/local/lib/python3.12/dist-packages/` (breaks on Mac, venv, Python 3.13 coming 2027, Docker slim images). `vllm.__file__` is canonical universal.

### Not yet done (Phase 2 target)

- Full monkey-patch glue from `kernels/` to upstream vllm modules (current v5.14.1 does text-replacement; v7.0 will use function-level monkey-patching via `patches/apply_all.py`)
- Remaining kernel implementations: `dequant_buffer.py`, `gdn_dual_stream.py`, `marlin_tuning.py`, `fp8_dispatcher.py`
- Test suites for the 4 remaining kernels
- Integration platform matrix tests (`test_platform_matrix.py`)
- Migration of Patches 1-25 from monolithic `patch_genesis_unified.py` to per-patch modular entries

## Late v7.0-dev additions (2026-04-24, session 2)

### New patches wired

- **P7** — GDN dual-stream in_proj parallelism. Text-patch on
  `model_executor/layers/mamba/gdn_linear_attn.py:544-545` replacing the
  serial `in_proj_qkvz` + `in_proj_ba` calls with a
  `DualStreamDispatcher.maybe_parallel(...)` call. Platform-safe:
  sequential fallback on CPU / XPU; true parallelism on CUDA SM ≥ 8.0.
- **P12** — Qwen3 `<tool_call>` as implicit reasoning end (ADDITIVE scope
  to avoid conflict with P27). Adds `_tool_call_token_id`,
  `is_reasoning_end`, `is_reasoning_end_streaming`,
  `extract_content_ids` methods to `Qwen3ReasoningParser`.
- **P24** — Per-SM auto-select for Marlin MoE `num_warps` and
  `num_stages`. Ampere A5000 (SM 8.6) → warps=4, stages=3 measured
  optimum. Env `VLLM_MARLIN_MOE_NUM_WARPS`, `_NUM_STAGES` still override.
- **P26** — TurboQuant prefill output prealloc. Helper
  `TurboQuantBufferManager.get_or_create_prefill_output` +
  `layer._tq_prefill_output` attach. Kernel text-patch deferred to A/B
  benchmark.
- **P27** — Qwen3 reasoning parser BEFORE-THINK fallback. Captures text
  before `<think>` (previously dropped) and routes it to `content` in
  both streaming and non-streaming paths. Coexists with P12.
- **P28** — GDN `core_attn_out` prealloc via `GdnCoreAttnManager`.
  Correct P19 redo: allocation is profiler-visible via the manager's
  first `get_or_create` on the max-sized buffer, not lazy in forward.
  Text-patch on `gdn_linear_attn.py:569-575` with unique anchor
  (includes the preceding #28182 comment) so `forward_xpu`'s identical
  line is untouched.
- **P29** — Verified the qwen3coder tool parser already contains
  bounded-index guards in the v7.0 baseline (lines 609-616, 659-666,
  436-438). Registration is a no-op on the current image; re-emits if a
  future vLLM upgrade regresses.
- **P32 / P33** — `_cu_2` and `synth_seq_lens` preallocs bundled with
  P22. `TurboQuantBufferManager.get_or_create_cu_2` +
  `get_or_create_synth_seq_lens`; attached to the layer inside
  `ensure_turboquant_buffers`.
- **P5b** — Scaffolding for the future pad-smaller-to-max KV
  unification. `kernels/page_size_padded.py` helpers
  (`is_p5b_enabled`, `compute_real_page_size_bytes`, `clamp_to_real_shape`)
  behind `GENESIS_ENABLE_P5B=1`. Kernel text-patch intentionally not
  shipped.

### Infrastructure

- `benchmarks/harness/` — Part 11.1 pre-deploy gate runner:
  - `gsm8k_regression`, `quality_harness`, `long_context_oom`,
    `tgs_decode`, `offline_api_parity`, `cuda_graph_recapture`,
    `run_all`.
  - Standard JSON report format, P0/P1 tiering, aggregated `summary.json`.
  - Dataset stubs in `benchmarks/data/`.
- `docs/RUNBOOK.md` — steady-state ops, diagnostic probes, blue/green
  deploy, rollback, known gotchas.

### Patch registry size

- Session start: 16 registered patches.
- Session end: **23 registered patches** (+P7, +P12, +P24, +P26, +P27,
  +P29, +P32/P33, +P28, +P5b).

### Compatibility

- Python: 3.10+ (uses modern type hints and `from __future__ import annotations`)
- PyTorch: 2.10+ (compatible with 2.11 upgrade in v0.20.0)
- Transformers: v5.0+ (compatible with vLLM v0.19.1+ requirement)
- vLLM: 0.19+ (tested against 0.19.2rc1.dev8, targeting 0.20.0)

### Author

Sandermage(Sander)-Barzov Aleksandr — Ukraine, Odessa
