# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 79c — stale spec_token_ids cleanup for unscheduled requests.

Backport of upstream PR vllm-project/vllm#37629 (OPEN as of 2026-04-26).
Fixes #36906: EAGLE3 + async-scheduling crash under high concurrency
caused by stale `spec_token_ids = [-1, ...]` placeholders leaking into
F.embedding(), triggering CUDA device-side assert.

================================================================
WHAT THIS FIXES
================================================================

In V1 async scheduling, `_update_after_schedule` sets
`spec_token_ids = [-1] * num_spec` as placeholders for every decode
request. These get cleared when the request is successfully scheduled.

But under high concurrency, the scheduler's token budget can be
exhausted before visiting all running requests. Unvisited running
requests retain stale `spec_token_ids = [-1, -1, -1]`, which creates
this thrashing cycle:

1. Scheduler skips request (budget exhausted before reaching it)
2. Worker removes from persistent batch (unscheduled)
3. Next step: scheduler re-schedules with stale `spec_token_ids = [-1, -1, -1]`
4. Worker re-adds, writes -1 to `token_ids_cpu` via
   `update_req_spec_token_ids()`
5. Scatter in `_prepare_input_ids` skips it (not in `prev_req_id_to_index`)
6. -1 reaches `F.embedding()` → CUDA device-side assert
7. Repeat forever

Most visible on multimodal models (large image prompts consume
disproportionate token budget per prefill chunk), but PR's regression
test proves it's NOT multimodal-specific — text-only with
sufficient concurrency reproduces.

================================================================
GENESIS APPROACH
================================================================

Mirror upstream exactly. After main scheduling loop in `schedule()`,
add a cleanup pass that clears `spec_token_ids` for any running
request not present in `num_scheduled_tokens`. Anchor immediately
before the `# Record the LoRAs in scheduled_running_reqs` comment
(stable, unique landmark).

================================================================
COMPATIBILITY
================================================================

- Only matters for `--async-scheduling` + spec-decode + high concurrency
- Composes cleanly with P79b (async proposer-sync) — orthogonal location
- Composes cleanly with P79d (preempt async-discard) — orthogonal
- Direct value for Genesis prod (max_num_seqs=2, sync ngram): NONE.
  Single-user can't exhaust token budget. Useful only for high-concurrency
  multimodal users on async + EAGLE/MTP.

================================================================
ENV
================================================================

GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1

================================================================
RISK
================================================================

LOW — defensive cleanup. Clearing `spec_token_ids` for unscheduled
requests is what `_update_after_schedule` would do anyway after a
successful schedule. Only changes behavior when budget exhaustion
prevents the normal cleanup. Idempotent: re-clearing already-empty
list is no-op.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: vllm#37629 (PR author per upstream).
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

log = logging.getLogger("genesis.wiring.p79c_stale_spec_token_cleanup")

GENESIS_P79C_MARKER = "Genesis P79c stale spec_token_ids cleanup backport vllm#37629 v7.46"


# Anchor on the unique `# Record the LoRAs in scheduled_running_reqs` comment.
# Insert the cleanup pass immediately before it.

P79C_OLD = (
    "        # Record the LoRAs in scheduled_running_reqs\n"
    "        scheduled_loras: set[int] = set()\n"
)

P79C_NEW = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P79c backport vllm#37629 — v7.49 improvement]\n"
    "        # Clear stale `-1` placeholder spec_token_ids for unscheduled\n"
    "        # running requests. In async scheduling, _update_after_schedule\n"
    "        # writes spec_token_ids = [-1, -1, ...] as placeholder intent.\n"
    "        # If a running request is NOT scheduled this step (token budget\n"
    "        # exhausted before reaching it), stale -1 values persist and\n"
    "        # leak into F.embedding() → CUDA device-side assert. Fixes #36906.\n"
    "        #\n"
    "        # v7.49 improvement (after upstream review of vllm#38624 by\n"
    "        # njhill + emerging canonical fix vllm#40768): only clear when\n"
    "        # spec_token_ids is ALL placeholders (-1). Real draft tokens\n"
    "        # (positive int IDs from MTP/EAGLE/ngram) must NEVER be cleared\n"
    "        # — they're valid for the next attempt. Old version naively\n"
    "        # cleared both, risking MTP state corruption.\n"
    "        #\n"
    "        # Also: prev_step_scheduled_req_ids membership is the proper\n"
    "        # gate (per vllm#40768). If req was NOT in prev step, worker\n"
    "        # won't consume placeholders anyway — safe to clear. If it WAS,\n"
    "        # placeholders may be needed by async input prep.\n"
    "        # CREDIT: vllm#37629 (OPEN), with refinement from vllm#40768.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        _genesis_p79c_prev_ids = getattr(self, \"prev_step_scheduled_req_ids\", None)\n"
    "        for request in self.running:\n"
    "            if request.request_id in num_scheduled_tokens:\n"
    "                continue\n"
    "            if not request.spec_token_ids:\n"
    "                continue\n"
    "            # Only clear pure placeholder lists (all -1). Real draft\n"
    "            # tokens are positive ints and must be preserved.\n"
    "            if not all(t == -1 for t in request.spec_token_ids):\n"
    "                continue\n"
    "            # If req was in prev worker step, placeholders may still\n"
    "            # be consumed by async input prep — leave them alone.\n"
    "            if (\n"
    "                _genesis_p79c_prev_ids is not None\n"
    "                and request.request_id in _genesis_p79c_prev_ids\n"
    "            ):\n"
    "                continue\n"
    "            request.spec_token_ids = []\n"
    "\n"
    "        # Record the LoRAs in scheduled_running_reqs\n"
    "        scheduled_loras: set[int] = set()\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P79c v1/core/sched/scheduler.py — stale spec_token_ids cleanup",
        target_file=str(target),
        marker=GENESIS_P79C_MARKER,
        sub_patches=[
            TextPatch(
                name="p79c_stale_spec_token_cleanup",
                anchor=P79C_OLD,
                replacement=P79C_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P79c",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P79c — stale spec_token_ids cleanup."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P79c")
    log_decision("P79c", decision, reason)
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
        log.info("[P79c] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift check: if upstream merged, the cleanup pass appears
    # immediately before the LoRAs comment with text matching the PR.
    # We probe for the presence of the cleanup pattern WITHOUT our marker.
    upstream_marker = "request.request_id not in num_scheduled_tokens"
    if upstream_marker in content and "[Genesis P79c" not in content:
        return "skipped", (
            f"upstream cleanup pattern {upstream_marker!r} present without "
            "Genesis marker — upstream PR #37629 may have merged equivalent fix"
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
        "P79c applied: schedule() now clears spec_token_ids for unscheduled "
        "running requests. Prevents -1 placeholder leak under budget-exhausted "
        "high-concurrency on async + EAGLE/MTP. Backport of vllm#37629 (OPEN). "
        "Fixes #36906."
    )
