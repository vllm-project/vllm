# SPDX-License-Identifier: Apache-2.0
"""Text-patch primitives for Phase 3 wiring.

Why text-patches (vs pure monkey-patch)
----------------------------------------
Some upstream code sites are not cleanly monkey-patchable:

  - Raises inside a method body (e.g. arg_utils.py `if model_config.is_hybrid:
    raise NotImplementedError(...)`) — the whole method body would have to be
    re-defined in Python to bypass the raise, which would duplicate ~50 lines
    of upstream logic that might drift between vLLM versions.

  - Compile-time Triton kernel literals (e.g. `BLOCK_KV=4, num_warps=1` as
    immediate arguments to `@triton.jit`) — these are baked into kernel
    compilation and cannot be overridden at call site.

  - Control-flow that depends on local variable state only available inside
    the original function (no clean rebind point).

For those, a surgical text-replacement at plugin-register time is pragmatic:
small, targeted, verifiable, and far less invasive than a full method rewrite.

Why not the v5.14.1 monolith approach directly
-----------------------------------------------
The monolith had real-world issues that this module addresses:

  1. **Anchor drift = hard crash.** If upstream changed a single line, the
     patch returned False and the whole patcher `sys.exit(1)`, leading to
     container restart-loops (observed on prod 2026-04-24).

  2. **Interlocking sub-patches.** Multiple sub-patches of one feature shared
     state via sequential `_c = _c.replace(...)`; if one sub-patch's anchor
     was already modified by a prior container run, the rest silently failed
     in confusing ways.

  3. **No post-write verification.** Monolith wrote the new content back
     without re-reading to confirm the intended change persisted.

This module's contract
----------------------
- Each `TextPatcher` instance represents ONE coherent rewrite of ONE file.
- Anchor mismatch → returns TextPatchResult.SKIPPED with reason (never raises).
- Already-applied (marker present) → returns APPLIED (idempotency win).
- File missing / permission denied → returns SKIPPED.
- Write succeeds → re-reads and verifies marker is present, else FAILED.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("genesis.wiring.text_patch")


class TextPatchResult(Enum):
    APPLIED = "applied"        # File modified in this call
    IDEMPOTENT = "idempotent"  # Marker already present, nothing to do
    SKIPPED = "skipped"        # Anchor drift / not applicable — not an error
    FAILED = "failed"          # Unexpected condition, patch safety violated


@dataclass(frozen=True)
class TextPatchFailure:
    """Why a patch ended in non-APPLIED/IDEMPOTENT state."""
    reason: str
    detail: str = ""


@dataclass
class TextPatch:
    """A single anchor→replacement edit.

    Attributes:
      name: Short identifier for logs.
      anchor: Exact substring that must appear in the file (pre-patch).
      replacement: What to substitute for `anchor`.
      required: If True, failure of this sub-patch aborts the parent group.
                If False (default), sibling sub-patches still run.
    """
    name: str
    anchor: str
    replacement: str
    required: bool = False


@dataclass
class TextPatcher:
    """Apply a sequence of anchor→replacement edits to one target file.

    Attributes:
      patch_name: Stable human-readable identifier (e.g. "P4 TQ hybrid").
      target_file: Absolute path to file to patch.
      marker: Unique string that, once present in the file, indicates this
              patch has been applied. Used for idempotency.
      sub_patches: Ordered list of TextPatch edits. Applied in sequence.
      upstream_drift_markers: If any of these strings are present in the
              file, consider this patch OBSOLETE (upstream merged a fix).
              Returns SKIPPED with a clear message.
    """
    patch_name: str
    target_file: str
    marker: str
    sub_patches: list[TextPatch]
    upstream_drift_markers: list[str] = field(default_factory=list)

    def apply(self) -> tuple[TextPatchResult, Optional[TextPatchFailure]]:
        """Execute the patch. Returns (result, failure_info_if_not_ok).

        NEVER raises — returns SKIPPED/FAILED on any issue.
        """
        # Layer 1: file must exist and be readable.
        if not os.path.isfile(self.target_file):
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="target_file_missing",
                detail=f"{self.target_file} not found",
            )

        try:
            with open(self.target_file) as f:
                content = f.read()
        except (OSError, PermissionError) as e:
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="read_error", detail=str(e),
            )

        # Layer 2: idempotency — already applied?
        if self.marker in content:
            log.debug(
                "[%s] marker %r already present — idempotent skip",
                self.patch_name, self.marker,
            )
            return TextPatchResult.IDEMPOTENT, None

        # Layer 3: upstream merged?
        for m in self.upstream_drift_markers:
            if m in content:
                log.info(
                    "[%s] upstream marker %r detected — patch obsolete, skip",
                    self.patch_name, m,
                )
                return TextPatchResult.SKIPPED, TextPatchFailure(
                    reason="upstream_merged",
                    detail=f"marker {m!r} present",
                )

        # Layer 4: validate ALL anchors before applying ANY. If any required
        # anchor is missing, abort without writing.
        modified = content
        applied_patches: list[str] = []

        for sp in self.sub_patches:
            if sp.anchor not in modified:
                # Anchor drift: not a crash, but we must decide whether to abort.
                if sp.required:
                    return TextPatchResult.SKIPPED, TextPatchFailure(
                        reason="required_anchor_missing",
                        detail=f"sub-patch {sp.name!r}: anchor not found in file",
                    )
                log.info(
                    "[%s/%s] anchor not found — soft skip (sibling patches continue)",
                    self.patch_name, sp.name,
                )
                continue

            if modified.count(sp.anchor) != 1:
                return TextPatchResult.SKIPPED, TextPatchFailure(
                    reason="ambiguous_anchor",
                    detail=(
                        f"sub-patch {sp.name!r}: anchor appears "
                        f"{modified.count(sp.anchor)} times (expected 1)"
                    ),
                )

            modified = modified.replace(sp.anchor, sp.replacement, 1)
            applied_patches.append(sp.name)

        if not applied_patches:
            # All sub-patches missed — treat as skip, not failure.
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="no_applicable_sub_patches",
                detail="every sub-patch anchor absent — file may be post-upstream-fix",
            )

        # Layer 5: prepend marker comment so future runs see IDEMPOTENT.
        # Uses Python-style `#` comment.
        #
        # G-014 audit note (2026-05-02): TextPatcher is currently Python-
        # only. All Genesis text-patches target `.py` files in the vllm
        # tree; the `#` marker is syntactically valid there. If
        # TextPatcher is ever extended to non-Python targets (yaml/sh/
        # cuda etc.), `marker_line` must switch to a syntax appropriate
        # to the file extension. Until then: contract is "Python source
        # files only".
        marker_line = f"# [Genesis wiring marker: {self.marker}]\n"
        if not modified.startswith(marker_line):
            modified = marker_line + modified

        # Layer 6: write + verify.
        try:
            with open(self.target_file, "w") as f:
                f.write(modified)
        except (OSError, PermissionError) as e:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="write_error", detail=str(e),
            )

        # Re-read to confirm.
        try:
            with open(self.target_file) as f:
                reread = f.read()
        except (OSError, PermissionError) as e:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="reread_error", detail=str(e),
            )

        if self.marker not in reread:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="marker_not_persisted",
                detail="file write succeeded but marker absent on re-read",
            )

        log.info(
            "[%s] applied %d sub-patches: %s",
            self.patch_name, len(applied_patches), ", ".join(applied_patches),
        )
        return TextPatchResult.APPLIED, None


# ────────────────────────────────────────────────────────────────────────
# B2 — Shared wiring-result mapper
#
# Most wiring `apply()` functions follow the same skeleton:
#   1. Check dispatcher should_apply
#   2. Check vllm_install_root + resolve target
#   3. Run patcher.apply()
#   4. Translate (TextPatchResult, TextPatchFailure | None) into the
#      wiring contract: ("applied" | "skipped" | "failed", reason: str)
#
# Step 4 was duplicated across ~25 wiring modules with subtle drift —
# some forgot to handle SKIPPED / IDEMPOTENT and silently reported
# "applied" when the file was actually unchanged (caught by PN14 TDD
# 2026-04-29). This helper centralizes the mapping so future patches
# (and the legacy ones as they migrate) can call it instead of
# rolling their own if/elif/else.

def result_to_wiring_status(
    result: "TextPatchResult",
    failure: "TextPatchFailure | None",
    *,
    applied_message: str,
    patch_name: str,
) -> tuple[str, str]:
    """Translate a (TextPatchResult, TextPatchFailure) pair into the
    wiring `apply()` return contract: (status, reason).

    Args:
        result: TextPatchResult enum value from `patcher.apply()`
        failure: optional TextPatchFailure from same call
        applied_message: human-readable success message (only used when
            result == APPLIED)
        patch_name: patch identifier for inclusion in skip / failure
            reasons (typically `patcher.patch_name`)

    Returns:
        (status, reason) where status ∈ {"applied", "skipped", "failed"}.
        APPLIED → ("applied", applied_message)
        IDEMPOTENT → ("skipped", "<patch_name>: already applied (marker present)")
        SKIPPED → ("skipped", "<patch_name>: <failure.reason> — <failure.detail>")
        FAILED → ("failed", "<patch_name>: <failure.reason> (<failure.detail>)")
    """
    if result == TextPatchResult.APPLIED:
        return "applied", applied_message
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", f"{patch_name}: already applied (marker present)"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown_skip"
        detail = failure.detail if failure and failure.detail else None
        msg = f"{patch_name}: {reason}"
        if detail:
            msg += f" — {detail}"
        return "skipped", msg
    # TextPatchResult.FAILED
    reason = failure.reason if failure else "unknown"
    detail = failure.detail if failure and failure.detail else ""
    return "failed", f"{patch_name}: {reason} ({detail})"


# ═══════════════════════════════════════════════════════════════════════════
# MultiFilePatchTransaction — Audit A-03/A-04/A-05 fix (2026-05-05)
# ═══════════════════════════════════════════════════════════════════════════


class MultiFilePatchTransaction:
    """Two-phase commit for multi-file text-patches.

    Audit context (2026-05-05): TextPatcher is per-file atomic only.
    Multi-file wiring patches (PN52, PN58) iterate over patchers — if file
    1 succeeds and file 2 fails, file 1 stays modified, leaving system in
    partial state. PN52 docstring even falsely promised rollback.

    This class implements proper validate-all-then-write-all transaction:

      Phase 1 (DRY-RUN): for each TextPatcher, peek at target content +
        verify all anchors present + replacements would be unique. NO
        files modified.
      Phase 2 (COMMIT): only if all dry-runs passed → real apply on each
        in order. If a Phase 2 step still fails (rare race condition —
        file modified between dry-run and commit), best-effort rollback
        via marker-based reverse search on already-modified files.

    Usage in PN52 / PN58:
        def apply():
            txn = MultiFilePatchTransaction([
                _make_envs_patcher(),
                _make_abs_parser_patcher(),
                _make_basic_parser_patcher(),
                _make_struct_out_patcher(),
                _make_sched_patcher(),
            ])
            return txn.apply_or_skip()

    Returns:
      ("applied", "PN52 5/5 sub-patchers committed") — full success
      ("skipped", "PN52 dry-run failed: file 3 anchor not found ...") — atomic skip
      ("failed", "PN52 partial commit: file 2 wrote, file 3 race; rollback ...") — degraded
    """

    def __init__(self, patchers: list["TextPatcher"], name: str = "multi-file"):
        self.patchers = list(patchers)
        self.name = name

    def _dry_run(self) -> tuple[bool, str]:
        """Phase 1: validate all patchers without writing.

        Returns: (all_ok, reason_if_not).

        Audit P1.2 fix 2026-05-05 (genesis_deep_cross_audit): now ALSO
        validates anchor uniqueness (`content.count(anchor) == 1`) for
        required sub-patches AND simulates sequential preview through
        all sub-patches in declared order, so an early replacement that
        would invalidate a later anchor is caught at dry-run time
        instead of producing a partial state at commit-phase.
        """
        for i, patcher in enumerate(self.patchers):
            if patcher is None:
                return False, f"file {i}: patcher is None (file not found)"
            if not os.path.isfile(patcher.target_file):
                return False, f"file {i} ({patcher.target_file}): file missing"
            try:
                src = open(patcher.target_file, "r", encoding="utf-8").read()
            except Exception as e:
                return False, f"file {i}: read failed ({e})"
            # If marker already present → already-applied → OK (idempotent)
            if patcher.marker in src:
                continue
            # Audit P1.2 — sequential preview: walk sub-patches in order,
            # check each anchor presence + uniqueness in the simulated
            # post-prior-replacement state. Optional sub-patches still
            # allowed to be missing.
            preview = src
            for sp in patcher.sub_patches:
                if sp.anchor not in preview:
                    if sp.required:
                        return False, (
                            f"file {i} ({patcher.target_file}): "
                            f"required anchor for sub-patch '{sp.name}' not "
                            "found (post sequential preview)"
                        )
                    continue  # optional anchor missing — skip in preview
                # Anchor uniqueness: ambiguous_anchor would silently apply
                # to first-occurrence-only at commit time.
                count = preview.count(sp.anchor)
                if sp.required and count > 1:
                    return False, (
                        f"file {i} ({patcher.target_file}): required anchor "
                        f"for sub-patch '{sp.name}' is ambiguous "
                        f"(found {count} times) — TextPatcher would replace "
                        "only the first occurrence, leaving partial state"
                    )
                # Apply replacement to preview so subsequent sub-patches
                # see the post-replacement state.
                preview = preview.replace(sp.anchor, sp.replacement, 1)
        return True, ""

    def apply_or_skip(self) -> tuple[str, str]:
        """Two-phase commit: dry-run, then real apply.

        Atomic skip on dry-run failure. **True rollback** on Phase 2 race
        (audit G-POST-08 fix 2026-05-05) — pre-commit snapshots of every
        target file held in memory; on first commit-phase failure, all
        previously-modified files are restored byte-for-byte from snapshot.

        If a snapshot restore itself fails (filesystem error, permission
        change), the snapshot is written to ``<target>.genesis_rollback``
        next to the file as a manual recovery aid and a WARN is logged.
        """
        ok, reason = self._dry_run()
        if not ok:
            return "skipped", f"{self.name} dry-run failed: {reason}"

        # Phase 2a: snapshot ALL targets BEFORE any write. Held in memory
        # for byte-for-byte restore. Files on disk are not touched yet.
        snapshots: dict[int, tuple[str, str]] = {}
        for i, patcher in enumerate(self.patchers):
            try:
                with open(patcher.target_file, "r", encoding="utf-8") as fh:
                    snapshots[i] = (patcher.target_file, fh.read())
            except Exception as e:
                # Atomic skip — we never modified anything, so just bail
                return ("skipped",
                        f"{self.name} pre-commit snapshot failed for file {i} "
                        f"({patcher.target_file}): {e}")

        # Phase 2b: real commit, in order
        committed: list[tuple[int, "TextPatcher"]] = []
        for i, patcher in enumerate(self.patchers):
            try:
                result, failure = patcher.apply()
            except Exception as e:
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} raised "
                    f"{type(e).__name__}: {e}",
                )
            if result in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
                committed.append((i, patcher))
            elif result == TextPatchResult.SKIPPED:
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} skipped after dry-run "
                    f"passed (race condition?): "
                    f"{failure.reason if failure else '?'}",
                )
            else:  # FAILED
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} failed: "
                    f"{failure.reason if failure else '?'}",
                )

        return ("applied",
                f"{self.name} {len(committed)}/{len(self.patchers)} files "
                "committed atomically")

    def _rollback_and_fail(
        self,
        committed: list[tuple[int, "TextPatcher"]],
        snapshots: dict[int, tuple[str, str]],
        reason: str,
    ) -> tuple[str, str]:
        """Audit G-POST-08 fix 2026-05-05: TRUE rollback of any partially-
        committed files using the in-memory snapshots taken at Phase 2a.

        IDEMPOTENT-result files are not restored (they were already applied
        before this transaction started — restoring would unapply prior
        state belonging to a different transaction).

        On unrecoverable filesystem error during restore, the snapshot is
        written to ``<target>.genesis_rollback`` as a manual recovery aid
        so the operator never has to reconstruct the original file by hand.
        """
        # Detect IDEMPOTENT vs APPLIED so we don't unapply pre-existing state.
        # IDEMPOTENT means the marker was already present when apply() ran —
        # i.e. file was unchanged in this transaction.
        applied_indices: list[int] = []
        for i, patcher in committed:
            snap_path, snap_content = snapshots.get(i, ("", ""))
            try:
                with open(patcher.target_file, "r", encoding="utf-8") as fh:
                    current = fh.read()
            except Exception:
                # Can't even read it back — preserve snapshot to disk
                self._write_rollback_aid(snap_path, snap_content)
                continue
            if current != snap_content:
                applied_indices.append(i)

        restored: list[str] = []
        rollback_aids: list[str] = []
        for i in applied_indices:
            snap_path, snap_content = snapshots[i]
            try:
                # Atomic write via temp + rename to mirror TextPatcher's
                # commit semantics — no torn writes on crash mid-restore.
                tmp = snap_path + ".genesis_rollback.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(snap_content)
                os.replace(tmp, snap_path)
                restored.append(snap_path)
            except Exception as e:
                # Restore failed — leave the snapshot on disk as a manual
                # recovery aid (operator can `mv .genesis_rollback FILE`).
                aid = self._write_rollback_aid(snap_path, snap_content)
                if aid:
                    rollback_aids.append(f"{snap_path} → {aid} ({e})")

        notes = []
        if restored:
            notes.append(f"ROLLED BACK {len(restored)} file(s): "
                         + ", ".join(restored))
        if rollback_aids:
            notes.append("MANUAL RECOVERY NEEDED for: "
                         + "; ".join(rollback_aids))
        if not restored and not rollback_aids:
            notes.append("no files needed rollback (transaction failed before "
                         "any APPLIED write)")
        return ("failed", reason + "\n" + "\n".join(notes))

    @staticmethod
    def _write_rollback_aid(path: str, content: str) -> str | None:
        """Write a snapshot to ``<path>.genesis_rollback`` for manual recovery.
        Returns the aid path on success, None on failure."""
        if not path:
            return None
        aid = path + ".genesis_rollback"
        try:
            with open(aid, "w", encoding="utf-8") as fh:
                fh.write(content)
            return aid
        except Exception:
            return None
