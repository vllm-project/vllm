# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 86 — ngram batch_propose O(N*K) → O(N+K) backport.

Backport of [vllm#40876](https://github.com/vllm-project/vllm/pull/40876)
by `aaronagent`. Replaces the O(N*K) membership scan in
`NgramProposer.batch_propose` (where N=batch_size, K=number of valid
ngram requests in the batch) with an O(N+K) direct-fill loop.

Original code (O(N*K) due to `i in valid_ngram_requests` on a list):

    draft_token_ids: list[list[int]] = []
    ...
    for i in range(num_requests):
        if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
            draft_token_ids.append(
                self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
            )
        else:
            draft_token_ids.append([])

Patched (O(N+K) direct fill):

    draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]
    for i in valid_ngram_requests:
        num_drafts = self.valid_ngram_num_drafts[i]
        if num_drafts > 0:
            draft_token_ids[i] = self.valid_ngram_draft[i, :num_drafts].tolist()

================================================================
IMPACT FOR OUR WORKLOAD
================================================================

Genesis prod runs with `max_num_seqs=2` (single-user homelab) and
`prompt_lookup_min=8` (strict ngram). For N=2, K=2, the difference is
negligible (~ns). Higher-concurrency deployments (multi-user, batch
serving) see meaningful gains: at N=64 batch with K=32 valid ngram
requests, the saving is N*K - (N+K) = 64*32 - 96 = 1952 list
membership-check ops per batch step.

Status: opt-in via `GENESIS_ENABLE_P86=1`. Default OFF. **Recommended
default-ON** when upstream PR #40876 doesn't merge — algorithmic correct
fix, no behavioral change.

Compatibility
-------------
- Pure ngram method (`speculative-config method=ngram`)
- Strict-ngram (`prompt_lookup_min` >= 4)
- Adaptive K controller (P77) — orthogonal, both can be enabled
- MTP / EAGLE / DFlash — N/A (this fires only on ngram path)
- Default-OFF safe for all configs

Author backport: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: aaronagent (vllm#40876).
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

log = logging.getLogger("genesis.wiring.p86_ngram_batch_propose_linear")


GENESIS_P86_MARKER = "Genesis P86 ngram batch_propose O(N+K) backport (vllm#40876) v7.53"


P86_OLD = (
    "        draft_token_ids: list[list[int]] = []\n"
    "\n"
    "        # Only run batch propose if there are requests needing ngram proposals.\n"
)


P86_NEW_PART_1 = (
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P86 vllm#40876 backport] O(N*K) → O(N+K) direct-fill.\n"
    "        # The original `i in valid_ngram_requests` membership scan is\n"
    "        # O(K) per iteration. For N requests this is O(N*K). Direct-fill\n"
    "        # iterating valid_ngram_requests is O(N+K). Negligible at N≤2 but\n"
    "        # meaningful at high-concurrency deployments.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "\n"
    "        # Only run batch propose if there are requests needing ngram proposals.\n"
)

# We also need to replace the for loop at the END of batch_propose
P86_OLD_LOOP = (
    "        for i in range(num_requests):\n"
    "            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:\n"
    "                draft_token_ids.append(\n"
    "                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()\n"
    "                )\n"
    "            else:\n"
    "                draft_token_ids.append([])\n"
)

P86_NEW_LOOP = (
    "        # Build the output list directly, filling only the valid entries.\n"
    "        # `i in valid_ngram_requests` on a list is O(K), so the previous\n"
    "        # `for i in range(num_requests)` loop was O(N*K); this is O(N+K).\n"
    "        # [Genesis P86 vllm#40876 backport]\n"
    "        draft_token_ids: list[list[int]] = [[] for _ in range(num_requests)]\n"
    "        for i in valid_ngram_requests:\n"
    "            num_drafts = self.valid_ngram_num_drafts[i]\n"
    "            if num_drafts > 0:\n"
    "                draft_token_ids[i] = self.valid_ngram_draft[i, :num_drafts].tolist()\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/ngram_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P86 v1/spec_decode/ngram_proposer.py — ngram batch_propose "
            "O(N+K) direct-fill (vllm#40876)"
        ),
        target_file=str(target),
        marker=GENESIS_P86_MARKER,
        sub_patches=[
            TextPatch(
                name="p86_strip_eager_alloc",
                anchor=P86_OLD,
                replacement=P86_NEW_PART_1,
                required=True,
            ),
            TextPatch(
                name="p86_direct_fill_loop",
                anchor=P86_OLD_LOOP,
                replacement=P86_NEW_LOOP,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P86",
            # If upstream PR #40876 lands, the loop will look like our P86_NEW
            "for i in valid_ngram_requests:\n            num_drafts =",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P86 — ngram batch_propose O(N+K) backport."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P86")
    log_decision("P86", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/ngram_proposer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P86] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P86" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix (PR #40876 merged?)",
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
        "P86 applied: ngram batch_propose now uses O(N+K) direct-fill "
        "(was O(N*K) with `i in valid_ngram_requests` membership scan). "
        "Algorithmic improvement, no behavioral change. Negligible at N≤2; "
        "meaningful at high-concurrency deployments."
    )
