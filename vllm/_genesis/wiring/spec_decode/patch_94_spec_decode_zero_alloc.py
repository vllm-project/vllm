# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 94 — spec-decode prepare_next_token_ids_padded zero-alloc.

Backport of [vllm#41043](https://github.com/vllm-project/vllm/pull/41043)
by `wangluochao902` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

In `LLMBaseProposer.prepare_next_token_ids_padded` (called once per
spec-decode step for ALL spec methods — Eagle, MTP, ngram, draft model),
replaces a 4-step Python+numpy chain with a direct in-place loop:

Before (vanilla vLLM nightly fe9c3d6c5):
    seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
    self.backup_next_token_ids.np[:num_reqs] = np.array(
        [
            requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
            for i in range(num_reqs)
        ],
        dtype=np.int32,
    )

After (this patch):
    for i in range(num_reqs):
        self.backup_next_token_ids.np[i] = requests[
            gpu_input_batch.req_ids[i]
        ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)

Removes:
1. GPU→CPU sync via `.tolist()` (forces CUDA stream sync inside hot path)
2. List comprehension allocation (~num_reqs Python objects per step)
3. `np.array(...)` allocation + copy into pre-allocated buffer

PR author measured **P99 TPOT -9.3 %** on Llama-3.1-8B + Eagle3 TP=4
(though mean TPOT improvement was only 0.8 % — bigger win is on tail
latency, hence Genesis Issue should report P99 alongside mean).

For our MTP K=3 single-stream workload, expected wall-TPS gain is
+2-4 %; tighter CV (less variance from per-step Python overhead) is
the main qualitative improvement.

================================================================
SAFETY MODEL
================================================================

- Algorithmic identity: produces same `backup_next_token_ids` values as
  the original. Verified by inspection of PR #41043 diff.
- No GPU code changes; only Python-side hot-path simplification.
- Idempotent via marker; drift detection on the original `.tolist()` line
  (if upstream merges this PR, our patch detects the resulting absence
  of that line and skips).
- Default OFF; opt-in via `GENESIS_ENABLE_P94=1`.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#41043.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p94_spec_decode_zero_alloc")

GENESIS_P94_MARKER = (
    "Genesis P94 spec-decode prepare_next_token_ids_padded zero-alloc "
    "(vllm#41043) v7.62.7"
)


P94_OLD = (
    "        # Precompute get_token_id for when there is no valid next token\n"
    "        num_reqs = gpu_input_batch.num_reqs\n"
    "        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()\n"
    "        self.backup_next_token_ids.np[:num_reqs] = np.array(\n"
    "            [\n"
    "                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])\n"
    "                for i in range(num_reqs)\n"
    "            ],\n"
    "            dtype=np.int32,\n"
    "        )\n"
)

P94_NEW = (
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P94 vllm#41043 backport] Zero-alloc precompute of\n"
    "        # backup token IDs. Eliminates GPU->CPU .tolist() sync, list-\n"
    "        # comprehension Python objects, and np.array(...) allocation\n"
    "        # in the hot path. PR author measured P99 TPOT -9.3% on\n"
    "        # Eagle3; expected +2-4% wall TPS on our MTP K=3 path.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        num_reqs = gpu_input_batch.num_reqs\n"
    "        for i in range(num_reqs):\n"
    "            self.backup_next_token_ids.np[i] = requests[\n"
    "                gpu_input_batch.req_ids[i]\n"
    "            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/spec_decode/llm_base_proposer.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P94 v1/spec_decode/llm_base_proposer.py — zero-alloc "
            "prepare_next_token_ids_padded (vllm#41043)"
        ),
        target_file=str(target),
        marker=GENESIS_P94_MARKER,
        sub_patches=[
            TextPatch(
                name="p94_zero_alloc_backup_token_ids",
                anchor=P94_OLD,
                replacement=P94_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P94",
            # Upstream-side markers if PR #41043 (or equivalent) merges:
            "Precompute backup token IDs for discarded requests",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P94 — spec-decode prepare_next_token_ids_padded zero-alloc."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P94")
    log_decision("P94", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/llm_base_proposer.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P94] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #41043 (or equivalent) appears merged",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "P94 applied: prepare_next_token_ids_padded uses in-place loop "
        "instead of .tolist() + list-comp + np.array(). Removes GPU->CPU "
        "sync, list-comp allocation, np.array allocation. Algorithmic "
        "identity preserved; expected +2-4% wall TPS on MTP K=3 spec decode."
    )


def is_applied() -> bool:
    """Return True iff our marker is present in the target file."""
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
