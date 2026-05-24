# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 79b — async-scheduling × spec-decode proposer sync fix.

Backport of upstream PR vllm-project/vllm#40610 (OPEN as of 2026-04-26,
tracked from issue #40608). Fixes a happens-before ordering bug where
`prepare_inputs_event` is recorded too early — causing the next async
batch to mutate persistent state (block tables, batch metadata) while
the previous batch's spec-decode proposer is still reading it on GPU.

================================================================
WHAT THIS FIXES
================================================================

In the V1 async-scheduling path:

1. `execute_model()` runs forward pass + records `prepare_inputs_event`
2. `sample_tokens()` runs sampling AND spec-decode proposer GPU work
3. The next batch's `_update_states()` waits on `prepare_inputs_event`
   before mutating block tables / persistent batch state

The bug: `prepare_inputs_event` was recorded during input prep, NOT
after the proposer runs. So next batch could enter and start mutating
block_table_buffer / spec_decode_metadata while the previous batch's
proposer was still reading those tensors on GPU.

Symptoms (per upstream issue #40608):
- Nondeterministic instability in async + EAGLE/MTP/ngram_gpu
- Stale state usage during spec-decode proposer execution
- Hard to reproduce — concurrency-sensitive race

================================================================
GENESIS APPROACH
================================================================

PR #40610 makes `sample_tokens()` a thin wrapper that:
1. Calls the renamed `_sample_tokens_impl()` (original body)
2. In `finally:`, re-records `prepare_inputs_event` AFTER everything

We mirror upstream exactly via single TextPatch — wrap the existing
function header to call `_sample_tokens_impl` then re-record event.
The original function body is preserved intact (just renamed via the
duplicated `def` block).

================================================================
COMPATIBILITY
================================================================

- Activates only when `--async-scheduling` is enabled (otherwise
  `prepare_inputs_event` stays None and the new `record()` is no-op)
- Composes cleanly with our P67 (TQ multi-query kernel) — different
  code paths
- Composes cleanly with P79d (preempt async-discard) — different
  function entry points
- Direct value for Genesis prod (sync ngram): NONE — async path
  not engaged. Only useful for users on async + spec-decode.

================================================================
ENV
================================================================

GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=1

================================================================
RISK
================================================================

LOW — additive wrapper. Re-recording an event that's already been
recorded is harmless (overwrites timestamp). On non-async path,
`prepare_inputs_event is None` short-circuits the new code.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport of: vllm#40610 (gh PR author per upstream).
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

log = logging.getLogger("genesis.wiring.p79b_async_proposer_sync")

GENESIS_P79B_MARKER = "Genesis P79b async proposer-sync backport vllm#40610 v7.46"


# Anchor on the existing sample_tokens header + first body line.
# The signature in our pin is verified identical to upstream's pre-PR state.

P79B_OLD = (
    "    @torch.inference_mode\n"
    "    def sample_tokens(\n"
    "        self, grammar_output: \"GrammarOutput | None\"\n"
    "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:\n"
    "        if self.execute_model_state is None:\n"
)

P79B_NEW = (
    "    @torch.inference_mode\n"
    "    def sample_tokens(\n"
    "        self, grammar_output: \"GrammarOutput | None\"\n"
    "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P79b backport vllm#40610] Re-record prepare_inputs_event\n"
    "        # AFTER spec-decode proposer GPU work completes (not just after\n"
    "        # input prep). Fixes async-scheduling × spec-decode race where the\n"
    "        # next batch's _update_states could mutate block_table / persistent\n"
    "        # batch metadata while the previous batch's proposer was still\n"
    "        # reading those tensors on GPU. Symptoms: nondeterministic stale\n"
    "        # state usage on async + EAGLE/MTP/ngram_gpu paths.\n"
    "        # No-op on non-async path: prepare_inputs_event is None.\n"
    "        # CREDIT: vllm#40610 (OPEN at backport time, tracked from #40608)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        try:\n"
    "            return self._sample_tokens_impl(grammar_output)\n"
    "        finally:\n"
    "            if self.prepare_inputs_event is not None:\n"
    "                self.prepare_inputs_event.record()\n"
    "\n"
    "    def _sample_tokens_impl(\n"
    "        self, grammar_output: \"GrammarOutput | None\"\n"
    "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:\n"
    "        if self.execute_model_state is None:\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P79b v1/worker/gpu_model_runner.py — async proposer-sync backport",
        target_file=str(target),
        marker=GENESIS_P79B_MARKER,
        sub_patches=[
            TextPatch(
                name="p79b_async_proposer_sync",
                anchor=P79B_OLD,
                replacement=P79B_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P79b",
            # Detect upstream merge: post-merge `_sample_tokens_impl` exists
            # at module level. We probe via direct symbol search below.
            "_sample_tokens_impl",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P79b — async-scheduling × spec-decode proposer sync fix."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P79b")
    log_decision("P79b", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_model_runner.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P79b] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift check: if upstream merged, `_sample_tokens_impl` exists.
    # Distinguish "we already applied" (would have hit marker check above)
    # from "upstream merged equivalent" (no Genesis marker, but symbol present).
    if "_sample_tokens_impl" in content and "[Genesis P79b" not in content:
        return "skipped", (
            "_sample_tokens_impl symbol present without Genesis marker — "
            "upstream PR #40610 may have merged equivalent fix"
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
        "P79b applied: sample_tokens() now wraps _sample_tokens_impl with "
        "finally-clause re-recording prepare_inputs_event AFTER proposer GPU "
        "work. Fixes async × spec-decode race on EAGLE/MTP/ngram_gpu paths. "
        "Backport of vllm#40610 (OPEN, draft)."
    )
