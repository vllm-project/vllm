# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 14 — BlockTable tail zero-fill runtime hook.

Problem
-------
When a `block_table` row slot is reused by a shorter request after a longer
one, stale block IDs linger past `num_blocks_per_row[row_idx]`. FlashInfer's
`_copy_page_indices_kernel` can scan past `num_blocks_per_row`, reading KV
memory belonging to another concurrent live request. Silent divergent
output at `temperature=0` across retries — impossible to debug without
knowing the race exists.

Reference: vLLM PR [#39591](https://github.com/vllm-project/vllm/pull/39591),
issue [#39589](https://github.com/vllm-project/vllm/issues/39589).

Fix (runtime monkey-patch)
--------------------------
Wrap `BlockTable.append_row` and `BlockTable.move_row` with post-call
tail-zero using our `zero_block_table_tail` helper. The wrappers call the
original then zero the tail at the slots between the new `end` and
`max_num_blocks_per_req`.

Why runtime monkey-patch (not text-patch)
-----------------------------------------
  - BlockTable is a stable public class — method rebind is safe
  - Text-patching multi-method files is fragile (anchor drift)
  - Rebind survives workers naturally (our plugin re-applies per-process)

Platform compatibility
----------------------
All platforms — pure numpy/torch indexing, no vendor-specific code.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("genesis.wiring.p14_block_table")


_GENESIS_P14_APPEND_MARKER = "_genesis_p14_append_wrapped"
_GENESIS_P14_MOVE_MARKER = "_genesis_p14_move_wrapped"


def _import_block_table() -> Any | None:
    try:
        from vllm.v1.worker.block_table import BlockTable
        return BlockTable
    except ImportError as e:
        log.info("[Genesis P14] block_table module not importable: %s", e)
        return None
    except Exception as e:
        log.warning("[Genesis P14] unexpected import error: %s", e)
        return None


def apply() -> tuple[str, str]:
    """Wire tail-zero hook into BlockTable. Never raises."""
    cls = _import_block_table()
    if cls is None:
        return "skipped", "block_table module not available"

    already_wrapped = (
        getattr(getattr(cls, "append_row", None), _GENESIS_P14_APPEND_MARKER, False)
        and getattr(getattr(cls, "move_row", None), _GENESIS_P14_MOVE_MARKER, False)
    )
    if already_wrapped:
        return "applied", "already wrapped (idempotent — process already patched)"

    if not hasattr(cls, "append_row") or not hasattr(cls, "move_row"):
        return "skipped", (
            "BlockTable.append_row / move_row not both present "
            "(upstream may have refactored this class)"
        )

    try:
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail
    except Exception as e:
        return "failed", f"genesis kernel import failed: {e}"

    # ── wrap append_row ─────────────────────────────────────────────────
    orig_append = cls.append_row

    def _genesis_wrapped_append(self, block_ids, row_idx: int) -> None:
        """Genesis P14 wrapper — call original then zero the tail.

        Safe if `block_ids` is empty (original returns early; we also skip
        zeroing because nothing appended and the existing tail is still
        whatever the last append left — which is always zero under our
        invariant once P14 is active and every row has been touched).
        """
        orig_append(self, block_ids, row_idx)
        try:
            # After original: num_blocks_per_row[row_idx] is the new `end`.
            end = int(self.num_blocks_per_row[row_idx])
            zero_block_table_tail(
                self.block_table.np,
                row_idx=row_idx,
                end=end,
                max_per_req=self.max_num_blocks_per_req,
            )
        except Exception as e:
            log.warning(
                "[Genesis P14] tail-zero after append_row failed: %s "
                "(original append proceeded; only the zero step failed)",
                e,
            )

    setattr(_genesis_wrapped_append, _GENESIS_P14_APPEND_MARKER, True)
    setattr(_genesis_wrapped_append, "_genesis_p14_original_append", orig_append)
    cls.append_row = _genesis_wrapped_append

    # ── wrap move_row ──────────────────────────────────────────────────
    orig_move = cls.move_row

    def _genesis_wrapped_move(self, src: int, tgt: int) -> None:
        """Genesis P14 wrapper — call original then zero the target's tail."""
        orig_move(self, src, tgt)
        try:
            num_blocks = int(self.num_blocks_per_row[tgt])
            zero_block_table_tail(
                self.block_table.np,
                row_idx=tgt,
                end=num_blocks,
                max_per_req=self.max_num_blocks_per_req,
            )
        except Exception as e:
            log.warning(
                "[Genesis P14] tail-zero after move_row failed: %s",
                e,
            )

    setattr(_genesis_wrapped_move, _GENESIS_P14_MOVE_MARKER, True)
    setattr(_genesis_wrapped_move, "_genesis_p14_original_move", orig_move)
    cls.move_row = _genesis_wrapped_move

    log.info(
        "[Genesis P14] rebound BlockTable.append_row + move_row "
        "(tail-zero-fill active for concurrent-request safety)"
    )
    return "applied", "BlockTable.append_row + move_row wrapped (effective in this process)"


def is_applied() -> bool:
    cls = _import_block_table()
    if cls is None:
        return False
    return (
        getattr(getattr(cls, "append_row", None), _GENESIS_P14_APPEND_MARKER, False)
        and getattr(getattr(cls, "move_row", None), _GENESIS_P14_MOVE_MARKER, False)
    )


def revert() -> bool:
    """Restore both original methods. Tests only."""
    cls = _import_block_table()
    if cls is None:
        return False
    reverted_any = False
    append = getattr(cls, "append_row", None)
    if append and getattr(append, _GENESIS_P14_APPEND_MARKER, False):
        orig = getattr(append, "_genesis_p14_original_append", None)
        if orig is not None:
            cls.append_row = orig
            reverted_any = True
    move = getattr(cls, "move_row", None)
    if move and getattr(move, _GENESIS_P14_MOVE_MARKER, False):
        orig = getattr(move, "_genesis_p14_original_move", None)
        if orig is not None:
            cls.move_row = orig
            reverted_any = True
    return reverted_any
