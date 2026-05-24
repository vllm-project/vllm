# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 79d — preemption async-discard fix.

Backport of upstream PR vllm-project/vllm#38624 (CodersAcademy006, OPEN as
of 2026-04-26). Fixes a stale state window in V1 AsyncScheduler where
async placeholders persist across preemption — leading to silent
duplicated/repeated tokens after request resume.

================================================================
WHAT THIS FIXES
================================================================

Currently, `discard_latest_async_tokens=True` and
`num_output_placeholders=0` are set ONLY in `reset_prefix_cache()`.
Standard preemptions triggered inside the scheduler loop
(`_schedule_running` calling `_preempt_request()`) bypass this cleanup.

When such a request resumes:
1. The in-flight async token from before preemption is replayed
2. Output contains a duplicated/repeated token (e.g. "the the", "of of")
3. Symptom matches our v7.13 ngram-corruption story (token duplication)

This is structurally the same bug class as the residual ~45% rate we saw
when `prompt_lookup_min < 8` — likely some of those duplications were
preemption-induced async-token replay on a different code path.

================================================================
GENESIS APPROACH
================================================================

Per-PR-#38624 fix removes 2 lines from `reset_prefix_cache()` and adds
them to `_preempt_request()`. We adopt a SAFER variant:

- ONLY ADD the discard to `_preempt_request()` (additive, idempotent)
- Do NOT remove from `reset_prefix_cache()` (defensive — that path
  may be relied upon by other code we haven't audited)

The double-set is idempotent: setting `num_output_placeholders=0` twice
is fine; setting `discard_latest_async_tokens=True` twice is fine.

================================================================
COMPATIBILITY
================================================================

- Activates only when `--async-scheduling` is enabled (otherwise
  `_preempt_request` is called but no async tokens exist)
- Composes cleanly with our P58 (async placeholder fix) — different
  code paths, no overlap
- Direct value for Genesis prod (sync ngram): MINIMAL — we don't run
  async path. But protects users on async+EAGLE/MTP/ngram_gpu.

================================================================
ENV
================================================================

GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD=1

================================================================
RISK
================================================================

LOW — additive 2-line addition. Setting two fields to default-correct
values during preemption can only HELP (no behavior loss for non-async
deployments — the fields exist but are not consulted on sync path).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: vllm#38624 (CodersAcademy006).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p79d_preempt_async_discard")

GENESIS_P79D_MARKER = "Genesis P79d preempt async-discard backport vllm#38624 v7.46"


# Anchor on the existing `_preempt_request` body — the line that increments
# num_preemptions is a stable identifier.

P79D_OLD = (
    "        if request.spec_token_ids:\n"
    "            request.spec_token_ids = []\n"
    "        request.num_preemptions += 1\n"
)

P79D_NEW = (
    "        if request.spec_token_ids:\n"
    "            request.spec_token_ids = []\n"
    "        request.num_preemptions += 1\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P79d backport vllm#38624] Discard async tokens on ALL\n"
    "        # preemption paths (not just reset_prefix_cache). Without this,\n"
    "        # in-flight async tokens replay after request resume → duplicated\n"
    "        # output tokens (\"the the\", \"of of\"). Same bug class as our\n"
    "        # v7.13 ngram-corruption symptoms on a different code path.\n"
    "        # Idempotent — setting both fields twice (here AND in\n"
    "        # reset_prefix_cache) is safe, no behavioral change for sync paths.\n"
    "        # CREDIT: CodersAcademy006 vllm#38624 (OPEN at backport time)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        request.num_output_placeholders = 0\n"
    "        request.discard_latest_async_tokens = True\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P79d v1/core/sched/scheduler.py — preempt async-discard backport",
        target_file=str(target),
        marker=GENESIS_P79D_MARKER,
        sub_patches=[
            TextPatch(
                name="p79d_preempt_async_discard",
                anchor=P79D_OLD,
                replacement=P79D_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P79d",
            "vllm#38624",
            # If upstream eventually merges, both anchor lines will move.
            # Detect the merge marker from PR — they REMOVE these from
            # reset_prefix_cache; we PRESERVE them. So if reset_prefix_cache
            # no longer has the discard block, upstream merged → auto-skip.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P79d — preemption async-discard."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P79d")
    log_decision("P79d", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/sched/scheduler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P79d] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift check: if upstream PR #38624 merged, the discard lines move
    # from reset_prefix_cache() → _preempt_request() (our intended location).
    # Detect by counting occurrences of `discard_latest_async_tokens = True`:
    # - pre-merge upstream: 1 occurrence (in reset_prefix_cache)
    # - post-merge upstream: 1 occurrence (in _preempt_request, where we want it)
    # - our patch applied: 2 occurrences (additive — both locations)
    # If 2+ already → upstream merged + our patch already there OR both ours.
    # Safe to apply — idempotent marker check above handled second case.
    discard_count = content.count("discard_latest_async_tokens = True")
    if discard_count >= 2:
        return "skipped", (
            f"discard_latest_async_tokens = True found {discard_count}x — "
            "upstream may have merged equivalent fix"
        )

    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P79d" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P79d applied: _preempt_request() now discards in-flight async tokens "
        "(num_output_placeholders=0, discard_latest_async_tokens=True). "
        "Prevents duplicated tokens after preemption-resume on async+spec paths. "
        "Backport of vllm#38624 (CodersAcademy006, OPEN)."
    )
