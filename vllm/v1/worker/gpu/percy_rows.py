# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy W1: the worker's checked copy of the scheduler's block-table rows.

Each worker keeps, per (request, group), the block ids and gens it holds and
the row digest (`sum(row_digest_term(col, block_id, gen)) mod 2^64`). Every
SchedulerOutput is applied in a fixed order (msg_seq, gen updates, dropped
rows, then per request revokes BEFORE appends) and the resulting row length
and digest are compared with the scheduler's. A mismatch refuses that request
for the step without changing batch shapes, so tensor-parallel ranks, which
all receive the same message, refuse the same requests and stay in lockstep.

Pure CPU (numpy) so it is unit-testable without a GPU; the model runner turns
the per-request outcome into block-table writes.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.grant_table import (
    NULL_BLOCK_ID,
    NULL_GEN,
    normalize_gen_updates,
    row_digest,
    row_digest_term,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_M64 = (1 << 64) - 1


class _Row:
    """One (request, group) row as this worker holds it."""

    __slots__ = ("block_ids", "gens", "digest")

    def __init__(self) -> None:
        self.block_ids: list[int] = []
        # None once an entry arrived without a gen (an unchecked message):
        # the digest can no longer be maintained for this row.
        self.gens: list[int] | None = []
        self.digest = 0

    def append(self, block_ids: Sequence[int], gens: Sequence[int] | None) -> None:
        if not block_ids:
            return
        start = len(self.block_ids)
        self.block_ids.extend(block_ids)
        if self.gens is None or gens is None or len(gens) != len(block_ids):
            self.gens = None
            return
        cols = range(start, start + len(block_ids))
        self.digest = (self.digest + row_digest(cols, block_ids, gens)) & _M64
        self.gens.extend(gens)

    def revoke(self, col: int) -> bool:
        """Point `col` at the null block. False if the row has no such column
        (the scheduler and this worker disagree)."""
        if col >= len(self.block_ids):
            return False
        if self.gens is not None:
            old = row_digest_term(col, self.block_ids[col], self.gens[col])
            new = row_digest_term(col, NULL_BLOCK_ID, NULL_GEN)
            self.digest = (self.digest - old + new) & _M64
            self.gens[col] = NULL_GEN
        self.block_ids[col] = NULL_BLOCK_ID
        return True


@dataclass
class RowPlan:
    """How the block table must change for one request this step.

    `writes[group]` is None (no change) or `(start, block_ids)`: the row's
    entries from column `start` to its end, after revokes and appends. A
    single contiguous write per row keeps the staged-write kernel's
    one-write-per-row contract and orders revokes before appends.
    """

    refused: bool
    writes: list[tuple[int, list[int]] | None]
    row_lens: list[int]


@dataclass
class StepVerdict:
    """Outcome of verifying one SchedulerOutput."""

    # Requests whose rows this worker refuses this step (W1, I7); each is
    # reported in ModelRunnerOutput.row_check_failures.
    refused: set[str] = field(default_factory=set)
    # The whole step was refused (msg_seq gap or repeat).
    step_refused: bool = False
    # Percy W5: the block-id writes that passed the gen check, for the
    # zeroer and the CoW copy kernel (None: nothing to launch).
    zero_block_ids: list[int] | None = None
    block_copies: list | None = None
    # Block ids whose zero or copy this worker skipped this step.
    skipped_block_ids: set[int] = field(default_factory=set)


class RowVerifier:
    """The worker's checked mirror of every live request's rows, plus its
    CPU generation mirror of the main KV pool (`gen_mirror`, for the W5
    filters). The digest covers ids and gens, so it is the row check; the
    mirror is not compared against row gens, since a group may live in
    another pool (e.g. a HiSparse host tier) whose gens it does not track,
    and gen updates travel in the same seq-checked message.

    Percy W5 adds the writers that address blocks by id rather than through
    slot_mapping: zero and CoW copy lists are filtered against `gen_mirror`,
    and a recurrent-state row (`state_groups`) holding a block whose gen
    moved on since it was granted is refused, so its state indices read the
    null block, which the state kernels skip.

    Args:
        num_groups: Number of KV cache groups.
        num_blocks: Blocks in the main KV pool (size of the gen mirror).
        state_groups: Groups whose kernels address recurrent state by block
            id (Mamba, GDN, linear attention, short conv); all in the main
            pool.

    """

    def __init__(
        self, num_groups: int, num_blocks: int, state_groups: Iterable[int] = ()
    ) -> None:
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.state_groups = tuple(state_groups)
        self.gen_mirror = np.ones(num_blocks, dtype=np.uint32)
        if num_blocks:
            self.gen_mirror[NULL_BLOCK_ID] = NULL_GEN
        # Requests a skipped zero or copy touched this step: refused even if
        # (re-)sent in full, since their blocks did not get that write.
        self._step_refuse: set[str] = set()
        # Totals since start, for metrics and tests.
        self.num_skipped_zeros = 0
        self.num_skipped_copies = 0
        self.num_stale_state_refusals = 0
        self._rows: dict[str, list[_Row]] = {}
        # Requests refused on an earlier step: their rows stay pointed at the
        # null block until the scheduler drops or re-sends them in full.
        self._poisoned: set[str] = set()
        self._last_msg_seq = 0
        self._step_refused = False
        self.verdict = StepVerdict()

    # ---------------------------------------------------------------- step
    def begin_step(self, scheduler_output: "SchedulerOutput") -> StepVerdict:
        """Verify msg_seq, apply gen updates and drop finished/preempted
        rows. Call before any per-request method of the same step."""
        so = scheduler_output
        self.verdict = StepVerdict()
        self._step_refused = False
        seq = so.msg_seq
        if seq:
            if self._last_msg_seq and seq != self._last_msg_seq + 1:
                logger.error(
                    "Percy: msg_seq %d after %d: refusing the step and "
                    "resyncing every request",
                    seq,
                    self._last_msg_seq,
                )
                self._step_refused = True
                self.verdict.step_refused = True
                # Every row may have drifted: refuse each until re-sent.
                self._poisoned.update(self._rows)
            # A fresh worker takes the first checked seq as its baseline.
            self._last_msg_seq = seq

        if so.gen_updates and so.gen_updates[0]:
            # Same normalisation as the device table (G1), so both mirrors
            # stay equal: no null-block update, last gen wins per block.
            ids, gens, dropped = normalize_gen_updates(
                so.gen_updates[0], so.gen_updates[1], self.num_blocks
            )
            if dropped:
                logger.error(
                    "Percy: dropped %d gen updates for the null block or "
                    "out-of-range block ids",
                    dropped,
                )
            self.gen_mirror[ids] = gens

        self.drop(so.finished_req_ids)
        if so.preempted_req_ids:
            self.drop(so.preempted_req_ids)
        self._step_refuse = set()
        self._check_block_id_writes(so)
        return self.verdict

    # ------------------------------------------------------ block-id writes
    def _gen_ok(self, block_ids: np.ndarray, gens: np.ndarray) -> np.ndarray:
        in_pool = (block_ids >= 0) & (block_ids < self.num_blocks)
        ok = np.zeros(block_ids.shape, dtype=bool)
        ids = block_ids[in_pool]
        ok[in_pool] = (self.gen_mirror[ids] == gens[in_pool]) & (
            gens[in_pool] != NULL_GEN
        )
        return ok

    def _check_block_id_writes(self, so: "SchedulerOutput") -> None:
        """W5: keep the zeros and CoW copies whose message gens match this
        worker's mirror (the scheduler already ran check_writes on them).
        Lists without gens (hand-built or legacy outputs) are unchecked.
        Every rank holds the same mirror and message, so all skip alike."""
        verdict = self.verdict
        zero_ids = getattr(so, "new_block_ids_to_zero", None)
        zero_gens = getattr(so, "new_block_gens_to_zero", None)
        if zero_ids and zero_gens is not None and len(zero_gens) == len(zero_ids):
            ids = np.asarray(zero_ids, dtype=np.int64)
            ok = self._gen_ok(ids, np.asarray(zero_gens, dtype=np.int64) & 0xFFFFFFFF)
            verdict.zero_block_ids = ids[ok].tolist() or None
            skipped = ids[~ok].tolist()
            self.num_skipped_zeros += len(skipped)
            verdict.skipped_block_ids.update(skipped)
        else:
            verdict.zero_block_ids = list(zero_ids) if zero_ids else None

        copies = getattr(so, "kv_cache_block_copies", None)
        copy_gens = getattr(so, "kv_cache_block_copy_gens", None)
        if copies and copy_gens is not None and len(copy_gens) == len(copies):
            pairs = np.asarray([tuple(c) for c in copies], dtype=np.int64)
            gens = np.asarray(copy_gens, dtype=np.int64).reshape(-1, 2) & 0xFFFFFFFF
            # The destination is written; the source only read, but a stale
            # source would copy someone else's KV into the destination.
            ok = self._gen_ok(pairs[:, 0], gens[:, 0]) & self._gen_ok(
                pairs[:, 1], gens[:, 1]
            )
            verdict.block_copies = [c for c, keep in zip(copies, ok) if keep] or None
            skipped = [c for c, keep in zip(copies, ok) if not keep]
            self.num_skipped_copies += len(skipped)
            for copy in skipped:
                verdict.skipped_block_ids.update(int(b) for b in tuple(copy))
        else:
            verdict.block_copies = list(copies) if copies else None

        if verdict.skipped_block_ids:
            self._refuse_holders(so, verdict.skipped_block_ids)

    def _refuse_holders(self, so: "SchedulerOutput", block_ids: set[int]) -> None:
        """Refuse every request holding a block whose zero or copy was skipped:
        its row lacks a write the scheduler counted on. Rare path: O(rows)."""
        holders: set[str] = set()
        for req_id, rows in self._rows.items():
            if any(b in block_ids for row in rows for b in row.block_ids):
                holders.add(req_id)
        for new in so.scheduled_new_reqs:
            if any(b in block_ids for ids in new.block_ids for b in ids):
                holders.add(new.req_id)
        cached = so.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is not None and any(
                b in block_ids for ids in new_ids for b in ids
            ):
                holders.add(req_id)
        logger.warning(
            "Percy W5: skipped zero/copy of blocks %s (stale gen); refusing "
            "requests %s",
            sorted(block_ids),
            sorted(holders),
        )
        self._step_refuse |= holders
        # Holders not scheduled this step stay refused until re-sent.
        self._poisoned |= holders & self._rows.keys()

    def _stale_state(self, rows: list[_Row]) -> bool:
        """A state row holds a block whose gen changed since it was granted
        (a forced revoke, T7, sends no row revoke): its kernels must not
        touch that block."""
        for group in self.state_groups:
            row = rows[group]
            if row.gens is None or not row.block_ids:
                continue
            ids = np.asarray(row.block_ids, dtype=np.int64)
            live = (ids != NULL_BLOCK_ID) & (ids < self.num_blocks)
            if not live.any():
                continue
            gens = np.asarray(row.gens, dtype=np.int64)[live] & 0xFFFFFFFF
            if np.any(self.gen_mirror[ids[live]] != gens):
                return True
        return False

    def drop(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._rows.pop(req_id, None)
            self._poisoned.discard(req_id)

    def num_rows(self) -> int:
        return len(self._rows)

    def row_of(self, req_id: str, group: int) -> tuple[list[int], int] | None:
        rows = self._rows.get(req_id)
        if rows is None:
            return None
        return list(rows[group].block_ids), rows[group].digest

    # -------------------------------------------------------------- requests
    def add_new(
        self,
        req_id: str,
        block_ids: Sequence[Sequence[int]],
        block_gens: Sequence[Sequence[int]] = (),
        expected_row_len: Sequence[int] = (),
        expected_digest: Sequence[int] = (),
    ) -> RowPlan:
        """A request (re-)sent in full: replace its rows."""
        self._poisoned.discard(req_id)
        rows = []
        for group in range(self.num_groups):
            row = _Row()
            gens = block_gens[group] if block_gens else None
            row.append(list(block_ids[group]), gens)
            rows.append(row)
        self._rows[req_id] = rows
        ok = self._matches(rows, expected_row_len, expected_digest)
        writes: list[tuple[int, list[int]] | None] = [
            (0, list(r.block_ids)) for r in rows
        ]
        return self._finish(req_id, rows, ok, writes)

    def apply_cached(
        self,
        req_id: str,
        new_block_ids: Sequence[Sequence[int]] | None,
        new_block_gens: Sequence[Sequence[int]] | None = None,
        revoked: Iterable[tuple[int, int]] | None = None,
        expected_row_len: Sequence[int] = (),
        expected_digest: Sequence[int] = (),
    ) -> RowPlan:
        """Apply a running request's revokes, then appends, and verify."""
        rows = self._rows.get(req_id)
        if rows is None:
            # Unknown to this worker (dropped delta of a new request, or a
            # restarted worker): it cannot hold a correct row.
            rows = [_Row() for _ in range(self.num_groups)]
            self._rows[req_id] = rows
            known = False
        else:
            known = True
        ok = known
        first_dirty = [len(r.block_ids) for r in rows]
        for group, col in revoked or ():
            if not 0 <= group < self.num_groups or not rows[group].revoke(col):
                ok = False
                continue
            first_dirty[group] = min(first_dirty[group], col)
        if new_block_ids is not None:
            for group in range(self.num_groups):
                gens = new_block_gens[group] if new_block_gens else None
                rows[group].append(list(new_block_ids[group]), gens)
        if expected_row_len:
            ok = ok and self._matches(rows, expected_row_len, expected_digest)
        elif not known:
            # Unchecked message for a request this worker never saw: nothing
            # to compare against; behave as vLLM did (append only).
            ok = True
        writes: list[tuple[int, list[int]] | None] = []
        for group, row in enumerate(rows):
            start = first_dirty[group]
            writes.append(
                (start, row.block_ids[start:]) if start < len(row.block_ids) else None
            )
        return self._finish(req_id, rows, ok, writes)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _matches(
        rows: list[_Row],
        expected_row_len: Sequence[int],
        expected_digest: Sequence[int],
    ) -> bool:
        if not expected_row_len:
            return True
        if len(expected_row_len) != len(rows) or len(expected_digest) != len(rows):
            return False
        for row, length, digest in zip(rows, expected_row_len, expected_digest):
            if len(row.block_ids) != length:
                return False
            if row.gens is None:
                continue
            if row.digest != (int(digest) & _M64):
                return False
        return True

    def _finish(
        self,
        req_id: str,
        rows: list[_Row],
        ok: bool,
        writes: list[tuple[int, list[int]] | None],
    ) -> RowPlan:
        row_lens = [len(r.block_ids) for r in rows]
        refused = (
            self._step_refused
            or not ok
            or req_id in self._poisoned
            or req_id in self._step_refuse
        )
        if not refused and self.state_groups and self._stale_state(rows):
            logger.error(
                "Percy W5: request %s's state row holds a revoked block; refusing it",
                req_id,
            )
            self.num_stale_state_refusals += 1
            refused = True
        if refused:
            if req_id not in self._poisoned:
                logger.error(
                    "Percy: row check failed for request %s; refusing it", req_id
                )
            self._poisoned.add(req_id)
            self.verdict.refused.add(req_id)
            # Point the whole row at the null block (one write per group).
            writes = [(0, [NULL_BLOCK_ID] * n) if n else None for n in row_lens]
        return RowPlan(refused=refused, writes=writes, row_lens=row_lens)
