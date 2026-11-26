# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy M1: the checkable parts of each SchedulerOutput.

The scheduler keeps vLLM's delta format. On top of it, every dispatched
output carries a sequence number, the gens of the blocks it grants, the row
revokes the workers must apply before appending, and per (request, group)
the length and digest the worker's row must have afterwards. The digest is
`sum(row_digest_term(col, block_id, gen)) mod 2^64` over the row, so both
sides maintain it in O(delta) per step.

The scheduler-side mirror below replays exactly the deltas it sends (revokes,
then appends), so it describes what a correct worker holds.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.core.grant_table import (
    NULL_BLOCK_ID,
    NULL_GEN,
    row_digest,
    row_digest_term,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput, WriteWindow

_M64 = (1 << 64) - 1


class _Row:
    """What the worker holds for one (request, group)."""

    __slots__ = ("block_ids", "gens", "digest")

    def __init__(self) -> None:
        self.block_ids: list[int] = []
        self.gens: list[int] = []
        self.digest = 0

    def append(self, block_ids: list[int], gens: list[int]) -> None:
        if not block_ids:
            return
        start = len(self.block_ids)
        cols = range(start, start + len(block_ids))
        self.digest = (self.digest + row_digest(cols, block_ids, gens)) & _M64
        self.block_ids.extend(block_ids)
        self.gens.extend(gens)

    def revoke(self, col: int, block_id: int) -> bool:
        """Point `col` at the null block if it still holds `block_id`."""
        if col >= len(self.block_ids) or self.block_ids[col] != block_id:
            return False
        old = row_digest_term(col, block_id, self.gens[col])
        new = row_digest_term(col, NULL_BLOCK_ID, NULL_GEN)
        self.digest = (self.digest - old + new) & _M64
        self.block_ids[col] = NULL_BLOCK_ID
        self.gens[col] = NULL_GEN
        return True


class PercyMessageBuilder:
    """Adds the Percy M1 fields to each SchedulerOutput.

    Args:
        kv_cache_manager: The scheduler's KV cache manager.
        num_lookahead_tokens: Slots allocated past the scheduled tokens
            (drafter writes), included in the write windows.
        num_position_slack: Positions before `num_computed_tokens` a worker
            may still write (async scheduling with speculative decoding: the
            worker rolls back rejected drafts the scheduler counted).

    """

    def __init__(
        self,
        kv_cache_manager: "KVCacheManager",
        num_lookahead_tokens: int = 0,
        num_position_slack: int = 0,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.num_lookahead_tokens = num_lookahead_tokens
        self.num_position_slack = num_position_slack
        self.msg_seq = 0
        self._rows: dict[str, list[_Row]] = {}
        # req_id -> [(group, col, old_block_id, old_gen)] not yet sent.
        self._pending_revokes: defaultdict[str, list[tuple[int, int, int, int]]] = (
            defaultdict(list)
        )
        managers = kv_cache_manager.coordinator.single_type_managers
        main_pool = kv_cache_manager.block_pool
        self._grant_tables = [m.block_pool.grant_table for m in managers]
        self._block_sizes = [m.block_size for m in managers]
        # Groups whose KV writes go through slot_mapping, at the columns of
        # the written positions, into the main pool. Others (recurrent state,
        # cross-attention, host tiers) are covered by W5 instead.
        self._windowed = [
            m.uses_slot_mapping
            and m.writes_follow_positions
            and m.block_pool is main_pool
            for m in managers
        ]

    def num_tracked_rows(self) -> int:
        return len(self._rows)

    def row_of(self, req_id: str, group: int) -> tuple[list[int], int] | None:
        """(block_ids, digest) of a mirrored row, for tests and debugging."""
        rows = self._rows.get(req_id)
        if rows is None:
            return None
        return list(rows[group].block_ids), rows[group].digest

    def _gens(self, group: int, block_ids: list[int]) -> list[int]:
        if not block_ids:
            return []
        return self._grant_tables[group].gens_of(block_ids).tolist()

    def _new_rows(self, block_ids: tuple[list[int], ...]) -> list[_Row]:
        rows = []
        for group, ids in enumerate(block_ids):
            row = _Row()
            row.append(list(ids), self._gens(group, ids))
            rows.append(row)
        return rows

    def _write_window(
        self, req_id: str, num_computed_tokens: int, num_scheduled_tokens: int
    ) -> "tuple[WriteWindow | None, ...]":
        managers = self.kv_cache_manager.coordinator.single_type_managers
        windows: list[WriteWindow | None] = []
        start = max(0, num_computed_tokens - self.num_position_slack)
        end = num_computed_tokens + num_scheduled_tokens + self.num_lookahead_tokens
        for group, manager in enumerate(managers):
            if not self._windowed[group]:
                windows.append(None)
                continue
            block_size = self._block_sizes[group]
            blocks = manager.req_to_blocks.get(req_id, ())
            first = start // block_size
            last = min(len(blocks), cdiv(end, block_size))
            if first >= last:
                windows.append((first,))
                continue
            ids = np.fromiter(
                (b.block_id for b in blocks[first:last]),
                dtype=np.int64,
                count=last - first,
            )
            gens = self._grant_tables[group].gens_of(ids).astype(np.int64)
            # 0 is never a valid gen: a write aimed at a null column fails.
            gens[ids == NULL_BLOCK_ID] = 0
            windows.append((first, *gens.tolist()))
        return tuple(windows)

    def annotate(self, scheduler_output: "SchedulerOutput") -> None:
        """Fill the Percy fields of an output the scheduler will dispatch."""
        so = scheduler_output
        self.msg_seq += 1
        so.msg_seq = self.msg_seq

        for (
            req_id,
            group,
            col,
            block_id,
            gen,
        ) in self.kv_cache_manager.take_row_revokes():
            self._pending_revokes[req_id].append((group, col, block_id, gen))

        # Workers drop finished and preempted requests before adding new ones.
        for req_ids in (so.finished_req_ids, so.preempted_req_ids or ()):
            for req_id in req_ids:
                self._rows.pop(req_id, None)
                self._pending_revokes.pop(req_id, None)

        num_scheduled = so.num_scheduled_tokens
        for new in so.scheduled_new_reqs:
            # A full row supersedes any revoke recorded before it.
            self._pending_revokes.pop(new.req_id, None)
            rows = self._new_rows(new.block_ids)
            self._rows[new.req_id] = rows
            new.block_gens = tuple(r.gens[:] for r in rows)
            new.expected_row_len = tuple(len(r.block_ids) for r in rows)
            new.row_digest = tuple(r.digest for r in rows)
            new.write_window = self._write_window(
                new.req_id,
                new.num_computed_tokens,
                num_scheduled.get(new.req_id, 0),
            )

        cached = so.scheduled_cached_reqs
        cached.new_block_gens = []
        cached.revoked = []
        cached.expected_row_len = []
        cached.row_digest = []
        cached.write_windows = []
        for i, req_id in enumerate(cached.req_ids):
            rows = self._rows.get(req_id)
            new_block_ids = cached.new_block_ids[i]
            if rows is None:
                # Never sent as a new request (hand-built test outputs).
                cached.new_block_gens.append(None)
                cached.revoked.append(None)
                cached.expected_row_len.append(())
                cached.row_digest.append(())
                cached.write_windows.append(())
                continue
            revoked: list[tuple[int, int]] = []
            for group, col, block_id, _ in self._pending_revokes.pop(req_id, ()):
                # A column past the row was nulled before the worker got it:
                # the appended ids already carry the null block.
                if rows[group].revoke(col, block_id):
                    revoked.append((group, col))
            gens: tuple[list[int], ...] | None = None
            if new_block_ids is not None:
                gens = tuple(
                    self._gens(group, ids) for group, ids in enumerate(new_block_ids)
                )
                for row, ids, g in zip(rows, new_block_ids, gens):
                    row.append(list(ids), g)
            cached.new_block_gens.append(gens)
            cached.revoked.append(tuple(revoked) or None)
            cached.expected_row_len.append(tuple(len(r.block_ids) for r in rows))
            cached.row_digest.append(tuple(r.digest for r in rows))
            cached.write_windows.append(
                self._write_window(
                    req_id,
                    cached.num_computed_tokens[i],
                    num_scheduled.get(req_id, 0),
                )
            )

        main_table = self.kv_cache_manager.grant_table
        if so.new_block_ids_to_zero:
            so.new_block_gens_to_zero = main_table.gens_of(
                so.new_block_ids_to_zero
            ).tolist()
        if so.kv_cache_block_copies:
            src = main_table.gens_of([c.src_block_id for c in so.kv_cache_block_copies])
            dst = main_table.gens_of([c.dst_block_id for c in so.kv_cache_block_copies])
            so.kv_cache_block_copy_gens = list(zip(src.tolist(), dst.tolist()))
