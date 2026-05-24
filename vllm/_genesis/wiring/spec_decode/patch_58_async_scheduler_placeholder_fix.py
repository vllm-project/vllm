# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 58 — async-scheduler -1 placeholder leakage fix.

Backport of vllm-project/vllm#40768 (z1ying, OPEN at time of writing).

================================================================
ROOT CAUSE OF #40831 / #40807 / #40756 (high confidence 2026-04-25)
================================================================

After three independent isolation passes (noonghunna's 6-probe ladder,
our P56/P57 buffer/routing experiments, our Probe 9 kernel determinism
check) all converged on "captured CUDA graph references stale state
outside the kernel buffers", a systematic upstream search located the
actual mechanism: the AsyncScheduler ships `[-1] * num_spec_tokens` as
a shared list reference for every request every step. Worker-side
`gpu_model_runner._prepare_input_ids` is supposed to overwrite those
`-1`s on GPU, but the overwrite path skips when `prev_positions[i] < 0`
(i.e. the request was not in the previous worker batch).

For newly-scheduled requests (after preemption, after a chunked-prefill
boundary, on long-context turns where requests churn) the `-1`s reach
the GPU. With cudagraph capture, the captured forward executes embedding
lookup with `-1` indices — either crashing (V100 IMA path of #37159 /
#40756) or silently returning garbage that propagates as a degenerate
token loop (our #40831 / noonghunna's reproductions on Qwen3.6-27B).

Why this matches every isolation result we have:

  Probe 1-3 (TQ-only configs)      → bug persists with TQ + spec
                                       (placeholder leakage triggered by
                                        spec-decode regardless of TQ)
  Probe 4 (your routing patch)     → doesn't fully close it
                                       (routing layer is below scheduler)
  Probe 6 (cudagraph_mode=NONE)    → works
                                       (no captured graph means embedding
                                        lookup runs after worker-side
                                        prep this step, with current
                                        buffer contents — no stale snap)
  Probe 8 (#40798 backport)        → bug persists
                                       (#40798 is workspace mgmt, not
                                        scheduler placeholder logic)
  Probe 9 (kernel bit-equality)    → deterministic
                                       (kernel is fine; data is `-1`)

All four "remaining open" hypotheses from our previous analysis (cudagraph
references stale state / dispatcher selects wrong piece / spec-decode
metadata mismatch / per-layer pointer drift) are downstream symptoms of
the same root cause: the scheduler shipped `-1`s that should never have
crossed the host→device boundary for these requests.

Fix design (from vllm#40768)
----------------------------
Three coordinated changes:

  1. `vllm/v1/request.py`: add `num_pending_async_spec_placeholders: int`
     field. `num_tokens_with_spec` includes it for budget accounting.

  2. `vllm/v1/core/sched/async_scheduler.py`: stop assigning the shared
     list reference. Instead set the new counter field. The counter
     captures *intent* to ship placeholders, not the placeholders
     themselves.

  3. `vllm/v1/core/sched/scheduler.py`: add `_consume_spec_decode_tokens_
     for_step()` method that materializes `[-1] * count` ONLY when the
     request is in `prev_step_scheduled_req_ids`. Otherwise drop entirely
     so no `-1` reaches the worker. Also clear the counter on preemption
     and on `update_draft_token_ids`.

Status: opt-in (`GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1`).

Compatibility:
  - Idempotent (marker check).
  - Anchor-drift safe (each sub-patch independently anchored; required
    sub-patches abort the group cleanly on miss).
  - Becomes auto-no-op once #40768 lands upstream (`upstream_drift_markers`
    detects the new method name and skips).
  - Touches three files; if any one anchor drifts the whole patch group
    is skipped (we never want a half-patched scheduler).

Risks acknowledged:
  - This is a scheduler-level change. Bug class affects ALL spec-decode
    workloads under async scheduling, not just TurboQuant. If the patch
    misfires it could break non-TQ spec-decode too.
  - Unit test `tests/test_p58_async_placeholder_fix.py` validates the
    failure-and-fix scenario in isolation.
  - First production deploy SHOULD be tested with our regression smoke
    suite (Genesis verify-full.sh) before serving real traffic.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Investigation supported by AI tooling for source navigation.
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

log = logging.getLogger("genesis.wiring.p58_async_scheduler_placeholder_fix")

GENESIS_P58_MARKER = "Genesis P58 async-scheduler -1 placeholder fix v7.59_p62_compat"


def _is_enabled() -> bool:
    """Env-gate. Off by default — opt-in via:
    GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
    """
    return os.environ.get(
        "GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── File 1: vllm/v1/request.py ──────────────────────────────────────────────
#
# Add `num_pending_async_spec_placeholders` field to Request.__init__ and
# include it in `num_tokens_with_spec`.

REQUEST_FIELD_OLD = (
    "        self.spec_token_ids: list[int] = []\n"
    "        self.num_computed_tokens = 0"
)

REQUEST_FIELD_NEW = (
    "        self.spec_token_ids: list[int] = []\n"
    "        # [Genesis P58] Backport vllm#40768: track async placeholder intent\n"
    "        # as a count, not a list-reference. Only materialized to [-1, ...]\n"
    "        # by Scheduler._consume_spec_decode_tokens_for_step when the request\n"
    "        # was in prev_step_scheduled_req_ids (so worker-side overwrite is\n"
    "        # guaranteed). Distinct from spec_token_ids which holds real drafts.\n"
    "        self.num_pending_async_spec_placeholders = 0\n"
    "        self.num_computed_tokens = 0"
)

REQUEST_NUM_TOKENS_OLD = (
    "    def num_tokens_with_spec(self) -> int:\n"
    "        return len(self._all_token_ids) + len(self.spec_token_ids)"
)

REQUEST_NUM_TOKENS_NEW = (
    "    def num_tokens_with_spec(self) -> int:\n"
    "        # [Genesis P58] Include async placeholder count for budget accounting.\n"
    "        return (\n"
    "            len(self._all_token_ids)\n"
    "            + len(self.spec_token_ids)\n"
    "            + self.num_pending_async_spec_placeholders\n"
    "        )"
)


def _make_request_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/request.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P58 request.py — placeholder counter field",
        target_file=str(target),
        marker=GENESIS_P58_MARKER + " :: request.py",
        sub_patches=[
            TextPatch(
                name="p58_request_field",
                anchor=REQUEST_FIELD_OLD,
                replacement=REQUEST_FIELD_NEW,
                required=True,
            ),
            TextPatch(
                name="p58_request_num_tokens_with_spec",
                anchor=REQUEST_NUM_TOKENS_OLD,
                replacement=REQUEST_NUM_TOKENS_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["num_pending_async_spec_placeholders"],
    )


# ─── File 2: vllm/v1/core/sched/async_scheduler.py ──────────────────────────
#
# Replace `request.spec_token_ids = self._spec_token_placeholders` with
# the counter-based intent tracking. Leave the (now dead) field declaration
# alone — removing it cleanly is harder via TextPatcher and the harm of
# leaving it is just a few bytes of unused state per scheduler instance.

ASYNC_SCHED_OLD = (
    "            request.num_output_placeholders += 1 + cur_num_spec_tokens\n"
    "            # Add placeholders for the new draft/spec tokens.\n"
    "            # We will update the actual spec token ids in the worker process.\n"
    "            request.spec_token_ids = self._spec_token_placeholders"
)

ASYNC_SCHED_NEW = (
    "            request.num_output_placeholders += 1 + cur_num_spec_tokens\n"
    "            # [Genesis P58] Backport vllm#40768: track placeholder intent as\n"
    "            # count, not list-reference. Materialized to [-1, ...] only by\n"
    "            # Scheduler._consume_spec_decode_tokens_for_step and only when\n"
    "            # request is in prev_step_scheduled_req_ids — guaranteeing that\n"
    "            # worker-side _prepare_input_ids will overwrite the -1s on GPU.\n"
    "            # Without this gate, -1s leak through embedding lookup → token\n"
    "            # corruption (#40831) or vectorized_gather IMA (#37159, #40756).\n"
    "            if self.num_spec_tokens > 0:\n"
    "                request.num_pending_async_spec_placeholders = self.num_spec_tokens\n"
    "            else:\n"
    "                request.num_pending_async_spec_placeholders = 0"
)


def _make_async_sched_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/async_scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P58 async_scheduler.py — counter-based placeholder intent",
        target_file=str(target),
        marker=GENESIS_P58_MARKER + " :: async_scheduler.py",
        sub_patches=[
            TextPatch(
                name="p58_async_sched_assignment",
                anchor=ASYNC_SCHED_OLD,
                replacement=ASYNC_SCHED_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["num_pending_async_spec_placeholders"],
    )


# ─── File 3: vllm/v1/core/sched/scheduler.py ────────────────────────────────
#
# Three sub-patches:
#   3a. Replace the `if request.spec_token_ids:` block in `schedule()` with a
#       call to `_consume_spec_decode_tokens_for_step` that ALSO checks the
#       counter and gates placeholder materialization on prev-step membership.
#   3b. Add `_consume_spec_decode_tokens_for_step` method itself, after
#       `schedule()` and before `_build_kv_connector_meta`.
#   3c. Clear placeholder counter in `_preempt_request` and in
#       `update_draft_token_ids`.

SCHED_SPEC_BLOCK_OLD = (
    "            # Speculative decode related.\n"
    "            if request.spec_token_ids:\n"
    "                num_scheduled_spec_tokens = (\n"
    "                    num_new_tokens\n"
    "                    + request.num_computed_tokens\n"
    "                    - request.num_tokens\n"
    "                    - request.num_output_placeholders\n"
    "                )\n"
    "                if num_scheduled_spec_tokens > 0:\n"
    "                    spec_token_ids = request.spec_token_ids\n"
    "                    if len(spec_token_ids) > num_scheduled_spec_tokens:\n"
    "                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]\n"
    "                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids\n"
    "\n"
    "                # New spec tokens will be set in `update_draft_token_ids` before the\n"
    "                # next step when applicable.\n"
    "                request.spec_token_ids = []"
)

SCHED_SPEC_BLOCK_NEW = (
    "            # [Genesis P58] Backport vllm#40768: speculative decode related.\n"
    "            if (\n"
    "                request.spec_token_ids\n"
    "                or request.num_pending_async_spec_placeholders > 0\n"
    "            ):\n"
    "                spec_token_ids = self._consume_spec_decode_tokens_for_step(\n"
    "                    request, num_new_tokens\n"
    "                )\n"
    "                if spec_token_ids is not None:\n"
    "                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids\n"
    "\n"
    "                # New spec tokens will be set in `update_draft_token_ids` before the\n"
    "                # next step when applicable.\n"
    "                request.spec_token_ids = []"
)

# Insert the new method between `schedule()`'s `return scheduler_output` and
# `_build_kv_connector_meta`. Anchor on the unique 3-line transition.
SCHED_NEW_METHOD_OLD = (
    "            self._update_after_schedule(scheduler_output)\n"
    "        return scheduler_output\n"
    "\n"
    "    def _build_kv_connector_meta("
)

SCHED_NEW_METHOD_NEW = (
    "            self._update_after_schedule(scheduler_output)\n"
    "        return scheduler_output\n"
    "\n"
    "    def _consume_spec_decode_tokens_for_step(\n"
    "        self, request, num_new_tokens: int\n"
    "    ):\n"
    "        \"\"\"[Genesis P58] Backport vllm#40768.\n"
    "\n"
    "        Build this step's speculative tokens and consume pending intent.\n"
    "\n"
    "        request.spec_token_ids stores only real draft token IDs. Async\n"
    "        placeholder intent is tracked separately in\n"
    "        request.num_pending_async_spec_placeholders and materialized as\n"
    "        -1 tokens only when the request was present in the previous worker\n"
    "        batch (required for GPU-side overwrite in async input preparation).\n"
    "        \"\"\"\n"
    "        num_scheduled_spec_tokens = (\n"
    "            num_new_tokens\n"
    "            + request.num_computed_tokens\n"
    "            - request.num_tokens\n"
    "            - request.num_output_placeholders\n"
    "        )\n"
    "        if num_scheduled_spec_tokens <= 0:\n"
    "            request.num_pending_async_spec_placeholders = 0\n"
    "            return None\n"
    "\n"
    "        pending_placeholders = request.num_pending_async_spec_placeholders\n"
    "        request.num_pending_async_spec_placeholders = 0\n"
    "\n"
    "        if request.spec_token_ids:\n"
    "            spec_token_ids = request.spec_token_ids\n"
    "        elif (\n"
    "            self.scheduler_config.async_scheduling\n"
    "            and pending_placeholders > 0\n"
    "            and request.request_id in self.prev_step_scheduled_req_ids\n"
    "        ):\n"
    "            # Worker-side _prepare_input_ids will overwrite these -1s on GPU.\n"
    "            spec_token_ids = [-1] * pending_placeholders\n"
    "        else:\n"
    "            return None\n"
    "\n"
    "        if len(spec_token_ids) > num_scheduled_spec_tokens:\n"
    "            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]\n"
    "        return spec_token_ids\n"
    "\n"
    "    def _build_kv_connector_meta("
)

# Clear placeholder counter on preemption.
SCHED_PREEMPT_OLD = (
    "        if request.spec_token_ids:\n"
    "            request.spec_token_ids = []\n"
    "        request.num_preemptions += 1"
)

SCHED_PREEMPT_NEW = (
    "        if request.spec_token_ids:\n"
    "            request.spec_token_ids = []\n"
    "        # [Genesis P58] Backport vllm#40768.\n"
    "        request.num_pending_async_spec_placeholders = 0\n"
    "        request.num_preemptions += 1"
)

# Clear placeholder counter when real drafts arrive or prefill chunks land.
#
# 2026-04-28: SPLIT INTO TWO NARROW ANCHORS to coexist with P62.
# Old single SCHED_DRAFT_OLD/NEW anchor included the lines P62 rewrites
# (`should_advance(request)` block → `validate_tokens_reasoning_aware`).
# After P62 applies first, the old monolith anchor stops matching and the
# entire P58 patch group skips with `required_anchor_missing`. The fix:
# split P58's two zero-resets into independent narrow anchors that survive
# whether P62 ran or not.
#
# Site A anchor — surrounds the `if is_prefill_chunk:` ... `continue` block.
#   Both pre-P62 and post-P62 layouts contain this block unchanged.
#   Verified unique (1 occurrence each) on both layouts 2026-04-28.
SCHED_DRAFT_SITE_A_OLD = (
    "            if request.is_prefill_chunk:\n"
    "                # Ignore draft tokens for prefill chunks.\n"
    "                if request.spec_token_ids:\n"
    "                    request.spec_token_ids = []\n"
    "                continue"
)
SCHED_DRAFT_SITE_A_NEW = (
    "            if request.is_prefill_chunk:\n"
    "                # Ignore draft tokens for prefill chunks.\n"
    "                if request.spec_token_ids:\n"
    "                    request.spec_token_ids = []\n"
    "                # [Genesis P58] Backport vllm#40768.\n"
    "                request.num_pending_async_spec_placeholders = 0\n"
    "                continue"
)

# Site B anchor — uses the boundary between `update_from_output` (final line
#   `request.spec_token_ids = spec_token_ids`) and the next method
#   `update_draft_token_ids_in_output`. P62's sched_udti rewrites the lines
#   ABOVE the final assignment but keeps the assignment itself unchanged,
#   so this 3-line boundary anchor matches whether P62 ran or not.
#   Verified unique on both pre-P62 and post-P62 layouts 2026-04-28.
SCHED_DRAFT_SITE_B_OLD = (
    "            request.spec_token_ids = spec_token_ids\n"
    "\n"
    "    def update_draft_token_ids_in_output("
)
SCHED_DRAFT_SITE_B_NEW = (
    "            request.spec_token_ids = spec_token_ids\n"
    "            # [Genesis P58] Backport vllm#40768.\n"
    "            request.num_pending_async_spec_placeholders = 0\n"
    "\n"
    "    def update_draft_token_ids_in_output("
)


def _make_scheduler_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P58 scheduler.py — placeholder gating + new method",
        target_file=str(target),
        marker=GENESIS_P58_MARKER + " :: scheduler.py",
        sub_patches=[
            TextPatch(
                name="p58_sched_spec_block",
                anchor=SCHED_SPEC_BLOCK_OLD,
                replacement=SCHED_SPEC_BLOCK_NEW,
                required=True,
            ),
            TextPatch(
                name="p58_sched_new_method",
                anchor=SCHED_NEW_METHOD_OLD,
                replacement=SCHED_NEW_METHOD_NEW,
                required=True,
            ),
            TextPatch(
                name="p58_sched_preempt",
                anchor=SCHED_PREEMPT_OLD,
                replacement=SCHED_PREEMPT_NEW,
                required=True,
            ),
            # 2026-04-28 split into two narrow anchors so P58 coexists with P62.
            # See SCHED_DRAFT_SITE_A/B_OLD comments for design rationale.
            TextPatch(
                name="p58_sched_draft_site_a",
                anchor=SCHED_DRAFT_SITE_A_OLD,
                replacement=SCHED_DRAFT_SITE_A_NEW,
                required=True,
            ),
            TextPatch(
                name="p58_sched_draft_site_b",
                anchor=SCHED_DRAFT_SITE_B_OLD,
                replacement=SCHED_DRAFT_SITE_B_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["_consume_spec_decode_tokens_for_step"],
    )


def apply() -> tuple[str, str]:
    """Apply P58 wiring (3 files, all-or-nothing). Never raises.

    Each file's TextPatcher is independent: if any one file's anchor drifts
    or the file is missing we abort the WHOLE patch group (we do not want a
    half-patched scheduler).
    """
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P58")
    log_decision("P58", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = [
        _make_request_patcher(),
        _make_async_sched_patcher(),
        _make_scheduler_patcher(),
    ]
    if any(p is None for p in patchers):
        missing = [
            name for name, p in zip(
                ["request.py", "async_scheduler.py", "scheduler.py"], patchers
            ) if p is None
        ]
        return "skipped", f"target file(s) not found: {missing}"

    # First pass: dry-run all patchers' anchor-finds to abort cleanly on drift.
    # (TextPatcher.apply() already validates anchors before writing, but doing
    #  one file's apply then another's is risky if file 2 fails after file 1
    #  succeeds. We bias toward all-or-nothing by reading each file once first.)
    for p in patchers:
        if not os.path.isfile(p.target_file):
            return "skipped", f"target file disappeared: {p.target_file}"
        with open(p.target_file) as f:
            content = f.read()
        # Idempotent: marker present means already-patched on prior boot.
        if p.marker in content:
            continue
        # Upstream merged: skip the whole group cleanly.
        for m in p.upstream_drift_markers:
            if m in content:
                return (
                    "skipped",
                    f"upstream drift marker {m!r} found in {p.target_file} — "
                    "vllm#40768 likely already merged or backported; nothing "
                    "to do.",
                )
        # Anchor presence check.
        for sp in p.sub_patches:
            if sp.required and sp.anchor not in content:
                return (
                    "skipped",
                    f"required anchor for {sp.name!r} not found in "
                    f"{p.target_file} — upstream code likely drifted; "
                    "P58 cannot apply safely without re-anchoring.",
                )

    # All anchors look healthy. Apply each patcher.
    results = []
    for p in patchers:
        result, failure = p.apply()
        if result == TextPatchResult.FAILED:
            return "failed", (
                f"{p.patch_name}: {failure.reason if failure else 'unknown'} "
                f"({failure.detail if failure else ''}) — partial state risk; "
                "container should be torn down (compose down + up -d) to "
                "restore from image baseline."
            )
        results.append((p.patch_name, result))

    applied = sum(1 for _, r in results if r == TextPatchResult.APPLIED)
    idempotent = sum(1 for _, r in results if r == TextPatchResult.IDEMPOTENT)
    skipped = sum(1 for _, r in results if r == TextPatchResult.SKIPPED)

    if skipped > 0:
        return "skipped", (
            f"{skipped} of 3 sub-patchers skipped — likely upstream drift on "
            f"some files. {applied} applied + {idempotent} idempotent. "
            "Recommend `compose down && up -d` and re-investigate anchors."
        )

    if applied + idempotent == 3:
        return "applied", (
            f"P58 backport applied: {applied} files modified, {idempotent} "
            "already-applied. Async scheduler -1 placeholder leakage fixed; "
            "spec-decode + cudagraph workloads should no longer loop or IMA. "
            "Validate with our regression smoke suite before serving traffic."
        )
    return "failed", (
        f"unexpected result combination: applied={applied}, "
        f"idempotent={idempotent}, skipped={skipped}"
    )
