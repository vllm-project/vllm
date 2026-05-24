# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N34 — WorkspaceManager runtime lock relaxation.

================================================================
PN34 — companion to PN33 for the runtime decode path
================================================================

PN33 fixes the BOOT-TIME under-counting in `_dummy_sampler_run` so that
`profile_run` correctly reserves workspace for `num_speculative_tokens`
draft tokens. That closes the workspace-lock AssertionError class
during boot.

But the runtime decode path also has a workspace-lock failure mode:
`turboquant_attn.py:1350:_decode_attention` can request a workspace
size that exceeds the locked-at-warmup ceiling on rare paths
(continuation-prefill into long context, MTP K=3 + decode mid-stream).
The strict `WorkspaceManager._ensure_workspace_size` assertion in
`vllm/v1/worker/workspace.py` then crashes the engine.

PN34 relaxes that strict assertion to a one-shot WARNING + grow-anyway.
Behavior matches the pre-v0.20 path (workspace was just resized as
needed; the lock added the assertion at the Python boundary).

================================================================
WHY THIS IS SAFE
================================================================

- The strict lock is a debugging aid, not a correctness guarantee.
  The underlying allocator was always allowed to grow workspace; the
  lock only added an assertion at the Python boundary.

- Genesis P98 has a similar shape but its anchor doesn't match v0.20
  (auto-skips with a v7.5 marker that's no longer valid). PN34 covers
  the v0.20+ codepath that P98 was supposed to revert.

- Sandermage's PROD configs (35B-A3B-FP8, 27B Lorbus fp8_e5m2) don't
  exercise the locked-grow path so they don't notice the assertion;
  TurboQuant + MTP K=3 configs (especially noonghunna's 27B Lorbus +
  TQ3) do.

================================================================
PORT CREDIT
================================================================

Direct port of `noonghunna/club-3090` setup-time sidecar
`patch_workspace_lock_disable.py` (commit 2b5ab4d / referenced at
docs/UPSTREAM.md). Promoted into Genesis as a first-class patch so
operators don't need a setup-time sidecar.

Long-term: vLLM core PR vllm#40706 (TQ scratch dedup + reserve worst
case at warmup) is the proper upstream fix. PN34 retires when that
lands.

Status: opt-in via GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1.
Default OFF — relaxes a strict-debug assertion, so should be
explicitly engaged by operators who need it.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pN34_workspace_lock_runtime_relax")


GENESIS_PN34_MARKER = (
    "Genesis PN34 WorkspaceManager runtime lock relaxation v7.68"
)


# Anchor: the strict AssertionError block in _ensure_workspace_size.
# Format mirrors vLLM v0.20.x source as observed on 2026-05-02.
PN34_ANCHOR = (
    "            if self._locked:\n"
    "                raise AssertionError(\n"
    "                    f\"Workspace is locked but allocation from "
    "'{get_caller_info()}' \"\n"
    "                    f\"requires {required_bytes / _MB:.2f} MB, "
    "current size is \"\n"
    "                    f\"{current_size / _MB:.2f} MB. \"\n"
    "                    \"Workspace growth is not allowed after locking.\"\n"
    "                )\n"
)


# Replacement: log one-shot warning at WARNING level + proceed with
# the existing grow path. The module-level flag prevents log spam if
# the grow path is exercised many times per request.
PN34_REPLACEMENT = (
    "            if self._locked:\n"
    "                # [Genesis PN34 v7.68] Relax strict workspace lock to\n"
    "                # WARN+grow. Companion to PN33 (boot-time warmup K-aware\n"
    "                # sizing) that covers the runtime decode path. See\n"
    "                # noonghunna club-3090 patch_workspace_lock_disable.py\n"
    "                # for the diagnosis. Future-retire on vllm#40706.\n"
    "                global _GENESIS_PN34_WORKSPACE_LOCK_WARNED\n"
    "                try:\n"
    "                    _genesis_pn34_already_warned = (\n"
    "                        _GENESIS_PN34_WORKSPACE_LOCK_WARNED\n"
    "                    )\n"
    "                except NameError:\n"
    "                    _genesis_pn34_already_warned = False\n"
    "                if not _genesis_pn34_already_warned:\n"
    "                    logger.warning(\n"
    "                        '[Genesis PN34] Workspace lock violated by %s '\n"
    "                        '(%.2f MB needed, %.2f MB sized) \\u2014 growing '\n"
    "                        'anyway. See vllm#40706 for upstream tracking. '\n"
    "                        'Future violations silenced.',\n"
    "                        get_caller_info(),\n"
    "                        required_bytes / _MB, current_size / _MB,\n"
    "                    )\n"
    "                    _GENESIS_PN34_WORKSPACE_LOCK_WARNED = True\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/workspace.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN34 v1/worker/workspace.py — strict lock relaxation "
            "(companion to PN33 boot-time fix; covers runtime decode path)"
        ),
        target_file=str(target),
        marker=GENESIS_PN34_MARKER,
        sub_patches=[
            TextPatch(
                name="pN34_workspace_lock_runtime_relax",
                anchor=PN34_ANCHOR,
                replacement=PN34_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN34",
            "_GENESIS_PN34_WORKSPACE_LOCK_WARNED",
            # noonghunna's setup-time sidecar marker — if their patch
            # already landed in the file, do not double-patch
            "LOCAL workspace lock disable",
            # vllm#40706 upstream landing markers (when it merges, the
            # file will gain different code paths and our anchor won't
            # match anyway, but extra safety)
            "reserve_turboquant_decode_workspace",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN34 — workspace lock runtime relaxation.

    Default OFF. Engage when:
      - PN33 is already on (boot-time warmup K-aware) AND
      - runtime decode `_decode_attention` still hits workspace_lock
        AssertionError on TurboQuant + spec-decode configs

    Disable via not setting `GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1`.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN34")
    log_decision("PN34", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/workspace.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN34] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— either external sidecar already applied OR upstream "
                "vllm#40706 (or equivalent) appears merged. Re-evaluate "
                "whether Genesis PN34 still adds value."
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
        "PN34 applied: WorkspaceManager strict lock relaxed to WARN+grow. "
        "Companion to PN33 (boot warmup) for the runtime decode path. "
        "Mirrors noonghunna's patch_workspace_lock_disable.py setup-time "
        "sidecar but landed in Genesis directly. Retires when vllm#40706 "
        "merges upstream."
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
