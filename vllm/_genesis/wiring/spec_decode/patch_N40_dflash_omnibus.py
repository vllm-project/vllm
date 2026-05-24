"""Wiring for PN40 — DFlash omnibus optimization.

v1 wires Sub-kernel A (fused per-layer K-norm) into
`vllm/model_executor/models/qwen3_dflash.py` `precompute_and_store_context_kv`
method, replacing the per-layer `ops.rms_norm` loop with a single fused
Triton kernel call.

Anchor target (qwen3_dflash.py, validated against pin 0.20.2rc1.dev9+g01d4d1ad3):

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        ops.rms_norm(
            all_k_normed[i],
            all_k[i],
            self._k_norm_weights[i],
            self._rms_norm_eps,
        )

Replacement uses PN40 fused_k_norm with strict no-regression fallback —
on any eligibility failure or kernel error, falls through to the
original per-layer loop (preserved verbatim under `else` branch).

Composition (no conflicts):
  - PN21 (DFlash SWA) — different file (dflash.py); unaffected
  - PN23 (combine_hidden_states cast) — same file, different method
  - PN24 (aux layer +1) — different file (gpu_model_runner.py)
  - PN37 (research artifact) — different code path (forward, not precompute)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging

# Audit A-19 (2026-05-05): tightly coupled subpatches — both apply
# or both stay un-applied. Shared marker is acceptable here because the
# subpatches together form one logical fix; partial application is not
# desired anyway. _AUDIT_A19_EXEMPT documents this intentional design.
_AUDIT_A19_EXEMPT = True  # tightly coupled subpatches
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN40_dflash_omnibus")

GENESIS_PN40_MARKER = "Genesis PN40 DFlash omnibus v1 (sub-kernel A: fused K-norm)"


# Anchor: the per-layer K-norm loop body. Multi-line exact match.
PN40_A_ANCHOR = (
    "        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---\n"
    "        all_k_normed = torch.empty_like(all_k)\n"
    "        for i in range(L):\n"
    "            ops.rms_norm(\n"
    "                all_k_normed[i],\n"
    "                all_k[i],\n"
    "                self._k_norm_weights[i],\n"
    "                self._rms_norm_eps,\n"
    "            )\n"
)

PN40_A_REPLACEMENT = (
    "        # --- [Genesis PN40-A] Fused per-layer RMSNorm K (single kernel) ---\n"
    "        # Strict-superset: PN40 fused_k_norm if eligible (Triton single launch),\n"
    "        # else falls through to per-layer ops.rms_norm loop (baseline).\n"
    "        all_k_normed = torch.empty_like(all_k)\n"
    "        try:\n"
    "            from vllm._genesis.kernels.pn40_dflash_omnibus import (\n"
    "                fused_k_norm as _genesis_pn40_fused_k_norm,\n"
    "                is_sub_a_eligible as _genesis_pn40_eligible,\n"
    "            )\n"
    "            # [Audit A-08 fix 2026-05-05] _k_norm_weights[i] may be a\n"
    "            # raw Tensor (not nn.Parameter wrapper) — use getattr fallback\n"
    "            # to avoid every-call AttributeError silenced by `except Exception`.\n"
    "            _genesis_pn40_w_stack = torch.stack(\n"
    "                [getattr(self._k_norm_weights[i], \"weight\", self._k_norm_weights[i])\n"
    "                 for i in range(L)], dim=0\n"
    "            )\n"
    "            if _genesis_pn40_eligible(all_k.shape, all_k.dtype):\n"
    "                _genesis_pn40_fused_k_norm(\n"
    "                    all_k, _genesis_pn40_w_stack,\n"
    "                    self._rms_norm_eps, out=all_k_normed,\n"
    "                )\n"
    "            else:\n"
    "                for i in range(L):\n"
    "                    ops.rms_norm(\n"
    "                        all_k_normed[i], all_k[i],\n"
    "                        self._k_norm_weights[i], self._rms_norm_eps,\n"
    "                    )\n"
    "        except Exception:  # noqa: BLE001  (defensive — never raise)\n"
    "            for i in range(L):\n"
    "                ops.rms_norm(\n"
    "                    all_k_normed[i], all_k[i],\n"
    "                    self._k_norm_weights[i], self._rms_norm_eps,\n"
    "                )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN40 qwen3_dflash.py — fused K-norm sub-kernel A",
        target_file=str(target),
        marker=GENESIS_PN40_MARKER,
        sub_patches=[
            TextPatch(
                name="pN40_a_fused_k_norm",
                anchor=PN40_A_ANCHOR,
                replacement=PN40_A_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN40",
            # If upstream lands a single fused per-layer K-norm op themselves
            "_fused_dflash_k_norm",
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# PN40 sub-C+D wiring: universal MTP/DFlash observability hook
# Targets scheduler.py update_from_output where num_accepted is computed.
# v1 observe-only (no runtime K override); feeds pn40_observe_accepted_len.
# ═══════════════════════════════════════════════════════════════════

GENESIS_PN40_SCHED_MARKER = (
    "Genesis PN40 sub-C+D scheduler observe hook v1 (universal MTP/DFlash)"
)

# Audit P1 fix 2026-05-05 (genesis_deep_cross_audit): the previous wiring
# used ONE TextPatcher with ONE shared marker for both subpatches
# (observe + k_trim). If only one anchor was found, the patcher applied
# only that subpatch and wrote the shared marker; on next boot the marker
# already present meant the second subpatch could never apply ("partial
# state forever"). To restore all-or-nothing semantics WITHOUT requiring
# the larger MultiFilePatchTransaction refactor, the two subpatches now
# live in separate TextPatchers with separate markers. They still share
# the same target file but each has its own observability + drift detection.
GENESIS_PN40_SCHED_OBSERVE_MARKER = (
    "Genesis PN40 sub-C+D scheduler observe hook v1.1 (per-subpatch marker)"
)
GENESIS_PN40_SCHED_K_TRIM_MARKER = (
    "Genesis PN40 sub-C scheduler k_trim hook v1.1 (per-subpatch marker)"
)

# Anchor: 4-line block where num_accepted is computed in update_from_output.
# Stable since vllm 0.20.x. We insert the observe call AFTER num_accepted is
# computed but BEFORE make_spec_decoding_stats so any sentinel-trip flag is
# visible to subsequent code paths in the same step.
PN40_SCHED_ANCHOR = (
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n"
    "                num_rejected = num_draft_tokens - num_accepted\n"
)

PN40_SCHED_REPLACEMENT = (
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n"
    "                num_rejected = num_draft_tokens - num_accepted\n"
    "                # [Genesis PN40 sub-C+D] universal observability hook\n"
    "                # Feeds accepted_len to adaptive K controller + stability\n"
    "                # sentinel. v1 observe-only (no runtime K override yet).\n"
    "                # Defensive: never raises (engine hot path).\n"
    "                try:\n"
    "                    from vllm._genesis.kernels.pn40_dflash_omnibus import (\n"
    "                        pn40_observe_accepted_len as _genesis_pn40_observe,\n"
    "                    )\n"
    "                    _genesis_pn40_spec_method = (\n"
    "                        getattr(\n"
    "                            getattr(self, 'vllm_config', None) or\n"
    "                            getattr(self, 'scheduler_config', None),\n"
    "                            'speculative_config', None,\n"
    "                        )\n"
    "                    )\n"
    "                    _genesis_pn40_method = (\n"
    "                        getattr(_genesis_pn40_spec_method, 'method', 'unknown')\n"
    "                        if _genesis_pn40_spec_method is not None else 'unknown'\n"
    "                    )\n"
    "                    _genesis_pn40_base_k = (\n"
    "                        getattr(\n"
    "                            _genesis_pn40_spec_method,\n"
    "                            'num_speculative_tokens', num_draft_tokens,\n"
    "                        )\n"
    "                        if _genesis_pn40_spec_method is not None else num_draft_tokens\n"
    "                    )\n"
    "                    _genesis_pn40_observe(\n"
    "                        _genesis_pn40_method, num_accepted,\n"
    "                        base_k=_genesis_pn40_base_k,\n"
    "                    )\n"
    "                except Exception:  # noqa: BLE001\n"
    "                    pass\n"
)


def _make_scheduler_observe_patcher() -> TextPatcher | None:
    """sub-C+D observability hook — separate marker (audit P1 fix)."""
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN40 scheduler.py — sub-C+D observability hook",
        target_file=str(target),
        marker=GENESIS_PN40_SCHED_OBSERVE_MARKER,
        sub_patches=[
            TextPatch(
                name="pN40_sched_observe",
                anchor=PN40_SCHED_ANCHOR,
                replacement=PN40_SCHED_REPLACEMENT,
                required=True,  # observe-only is the core PN40 contract
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN40",
            "_genesis_pn40_observe",
        ],
    )


def _make_scheduler_k_trim_patcher() -> TextPatcher | None:
    """sub-C runtime K-trim hook — separate marker (audit P1 fix)."""
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN40 scheduler.py — sub-C runtime K-trim hook",
        target_file=str(target),
        marker=GENESIS_PN40_SCHED_K_TRIM_MARKER,
        sub_patches=[
            TextPatch(
                name="pN40_sched_k_trim",
                anchor=PN40_K_TRIM_ANCHOR,
                replacement=PN40_K_TRIM_REPLACEMENT,
                required=False,  # graceful skip if upstream removed update_draft_token_ids site
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN40",
            "_genesis_pn40_trim",
        ],
    )


def _make_scheduler_patcher() -> TextPatcher | None:
    """Backwards-compat shim — kept for callers that haven't migrated yet.

    Returns the OBSERVE patcher; K-trim must be applied separately via
    `_make_scheduler_k_trim_patcher()`. The previous combined-marker form
    silently locked the second subpatch out (audit P1 fix 2026-05-05).
    """
    return _make_scheduler_observe_patcher()


# ═══════════════════════════════════════════════════════════════════
# PN40 sub-C runtime K-trim (partial runtime override)
# Targets scheduler.update_draft_token_ids — point where proposer's draft
# tokens are stored on request before next-step verify. If controller
# recommends K < base (acceptance dropped), trim spec_token_ids list to
# the recommended K. Saves verify compute (less spec tokens to FA pass)
# but NOT draft compute (drafter already finished).
# Full proposer-side override (skip drafting entirely when K=0) requires
# vllm spec-decode infrastructure rework — separate sprint.
# ═══════════════════════════════════════════════════════════════════

# Anchor: PRISTINE form (no Genesis hooks). PN40 sub-C K-trim applies
# BEFORE P62/P58 wiring patches because PN40 dispatcher entry comes
# earlier in apply_all order. The anchor pair `request.spec_token_ids =
# spec_token_ids` + blank line + `def update_draft_token_ids_in_output(`
# is unique in the file (one occurrence in pristine scheduler.py from
# vllm 0.20.x).
PN40_K_TRIM_ANCHOR = (
    "            request.spec_token_ids = spec_token_ids\n"
    "\n"
    "    def update_draft_token_ids_in_output(\n"
)

PN40_K_TRIM_REPLACEMENT = (
    "            # [Genesis PN40 sub-C runtime K-trim] If adaptive controller\n"
    "            # recommends K < base, trim spec_token_ids. Saves verify\n"
    "            # compute on later verify pass. Defensive: never raises;\n"
    "            # falls through to baseline if PN40 disabled or trip.\n"
    "            # NOTE: Inserted BEFORE P62/P58 hooks (which extend this\n"
    "            # area). Both patches compose additively — see PN40 design.\n"
    "            try:\n"
    "                from vllm._genesis.kernels.pn40_dflash_omnibus import (\n"
    "                    pn40_get_recommended_k as _genesis_pn40_get_k,\n"
    "                    env_enabled as _genesis_pn40_master_enabled,\n"
    "                    sub_c_enabled as _genesis_pn40_sub_c_enabled,\n"
    "                )\n"
    "                if (\n"
    "                    spec_token_ids\n"
    "                    and _genesis_pn40_master_enabled()\n"
    "                    and _genesis_pn40_sub_c_enabled()\n"
    "                ):\n"
    "                    _genesis_pn40_spec_cfg = (\n"
    "                        getattr(\n"
    "                            getattr(self, 'vllm_config', None) or\n"
    "                            getattr(self, 'scheduler_config', None),\n"
    "                            'speculative_config', None,\n"
    "                        )\n"
    "                    )\n"
    "                    _genesis_pn40_method = (\n"
    "                        getattr(_genesis_pn40_spec_cfg, 'method', 'unknown')\n"
    "                        if _genesis_pn40_spec_cfg is not None else 'unknown'\n"
    "                    )\n"
    "                    _genesis_pn40_base_k = len(spec_token_ids)\n"
    "                    # [v7.72 workload-aware] read classifier tag from request\n"
    "                    _genesis_pn40_wkld = getattr(\n"
    "                        request, '_genesis_pn40_workload_class', None,\n"
    "                    )\n"
    "                    _genesis_pn40_rec_k = _genesis_pn40_get_k(\n"
    "                        _genesis_pn40_method, _genesis_pn40_base_k,\n"
    "                        workload_class=_genesis_pn40_wkld,\n"
    "                    )\n"
    "                    if 0 <= _genesis_pn40_rec_k < _genesis_pn40_base_k:\n"
    "                        # _genesis_pn40_trim: shrink spec list to rec_k.\n"
    "                        spec_token_ids = spec_token_ids[:_genesis_pn40_rec_k]\n"
    "            except Exception:  # noqa: BLE001 — never break engine\n"
    "                pass\n"
    "            request.spec_token_ids = spec_token_ids\n"
    "\n"
    "    def update_draft_token_ids_in_output(\n"
)


def apply() -> tuple[str, str]:
    """Apply PN40 wiring (sub-A DFlash + sub-C+D scheduler observe hook).

    Returns aggregated status. A=DFlash-only, C+D=universal MTP+DFlash.
    Each sub-patch independently degrades to "skipped" if its target is
    missing — no cascading failure.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN40")
    log_decision("PN40", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    results = []

    # Sub-A: DFlash drafter K-norm fusion (DFlash-only, qwen3_dflash.py)
    a_patcher = _make_patcher()
    if a_patcher is not None and os.path.isfile(a_patcher.target_file):
        a_result, a_failure = a_patcher.apply()
        results.append(("sub-A", a_result, a_failure))
    else:
        results.append(("sub-A", None, "qwen3_dflash.py not resolvable (DFlash-only)"))

    # Sub-C+D: universal MTP/DFlash scheduler observe hook + sub-C K-trim.
    # Audit P1 fix 2026-05-05: split into two TextPatchers with separate
    # markers so partial-apply state can no longer block the other.
    obs_patcher = _make_scheduler_observe_patcher()
    if obs_patcher is not None and os.path.isfile(obs_patcher.target_file):
        s_result, s_failure = obs_patcher.apply()
        results.append(("sub-C+D-sched-observe", s_result, s_failure))
    else:
        results.append(("sub-C+D-sched-observe", None, "scheduler.py not resolvable"))

    trim_patcher = _make_scheduler_k_trim_patcher()
    if trim_patcher is not None and os.path.isfile(trim_patcher.target_file):
        t_result, t_failure = trim_patcher.apply()
        results.append(("sub-C-sched-k-trim", t_result, t_failure))
    else:
        results.append(("sub-C-sched-k-trim", None, "scheduler.py not resolvable"))

    # Sub-D: workload classifier hook (per-request entry point).
    # Lives in middleware/ — different file from sched + DFlash patches.
    try:
        from vllm._genesis.wiring.middleware import (
            patch_N40_workload_classifier_hook,
        )
        cls_status, cls_reason = patch_N40_workload_classifier_hook.apply()
        # Translate string → fake TextPatchResult for aggregation
        if cls_status == "applied":
            results.append(("sub-D-classifier", TextPatchResult.APPLIED, None))
        elif cls_status == "skipped":
            results.append(("sub-D-classifier", TextPatchResult.SKIPPED, cls_reason))
        else:
            results.append(("sub-D-classifier", TextPatchResult.FAILED, cls_reason))
    except Exception as e:
        results.append(("sub-D-classifier", None, f"classifier wiring import failed: {e}"))

    # Aggregate status: report which sub-patches landed.
    # Direct enum check (no kwargs needed for result_to_wiring_status).
    applied_subs = [
        name for (name, r, _) in results
        if r is not None and r in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT)
    ]
    skipped_subs = [
        (name, r, f) for (name, r, f) in results
        if r is None or r == TextPatchResult.SKIPPED
    ]
    failed_subs = [
        (name, f) for (name, r, f) in results
        if r == TextPatchResult.FAILED
    ]

    if failed_subs:
        return "failed", f"sub-patches failed: {failed_subs}"
    if not applied_subs:
        return "skipped", (
            "no sub-patches applied; reasons: "
            + ", ".join(f"{n}={f or r}" for (n, r, f) in skipped_subs)
        )
    # [Audit A-10 fix 2026-05-05] Honest partial-vs-full status.
    # Previously returned `applied` even when only 1/N subpatches landed —
    # masked partial state from operator. Now: `applied` only when ALL
    # sub-patches succeed, else `partial` (downstream still treats as ON).
    # NOTE: caller `apply_patch_N40_dflash_omnibus` must handle `partial`
    # status the same as `applied` (both mean: don't error, but log honestly).
    if skipped_subs:
        return "partial", (
            f"PN40 PARTIAL: {len(applied_subs)} of "
            f"{len(applied_subs) + len(skipped_subs)} sub-patches landed. "
            f"Applied: {applied_subs}. "
            f"Skipped: {[n for (n, _, _) in skipped_subs]}. "
            "(See dispatcher logs for skip reasons. Operator may need to "
            "verify whether skipped sub-patches' anchors drifted.)"
        )
    return "applied", (
        f"PN40 applied: {len(applied_subs)} sub-patch(es) landed: "
        f"{applied_subs}. Sub-A: per-layer K-norm fusion (DFlash, "
        "+37us/+70us per draft step on 27B/35B). Sub-C+D: universal "
        "MTP+DFlash observability hook (adaptive K controller + stability "
        "sentinel; v1 observe-only, no runtime override yet)."
    )
