# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step draft scheduling for the standalone runtime.

In co-located mode the target model runner hands a speculator its own step:
target positions and sequence lengths, the shared block tables, and how many
tokens were sampled and rejected per request. The standalone runtime has no
target runner, so the planner rebuilds exactly those fields for the draft's
own KV cache from what the protocol carries. A request's draft state mirrors
the target's computed tokens: rows rewrite KV for every position they cover,
including positions the target then rejects, and only the accepted prefix is
committed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool, DraftKVExhausted


@dataclass(frozen=True)
class DraftStepRow:
    """One sequence's share of a draft step.

    ``num_tokens`` target tokens starting at the sequence's committed length
    are covered; the last ``num_rejected`` of them were rejected by the
    target. ``num_sampled == 0`` marks a prompt chunk that is not the last:
    its final position is completed by ``next_prefill_tokens`` instead of a
    sampled token.
    """

    key: SequenceKey
    num_tokens: int
    num_sampled: int
    num_rejected: int = 0
    last_sampled: int = 0
    next_prefill_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockTableUpdate:
    """New blocks for one request slot's block table.

    ``start`` is the table index of the first new block; 0 means the slot
    belongs to a new sequence and its previous table must be discarded.
    """

    start: int
    block_ids: list[int]


@dataclass(frozen=True)
class DraftStepPlan:
    """Batch-ordered step inputs for the rows the planner admitted."""

    rows: tuple[DraftStepRow, ...]
    req_indices: np.ndarray
    """[num_reqs] request slot of each row (the runner's idx_mapping)."""
    num_scheduled_tokens: np.ndarray
    """[num_reqs] target tokens per row."""
    query_start_loc: np.ndarray
    """[num_reqs + 1] prefix sums of num_scheduled_tokens."""
    num_computed_tokens: np.ndarray
    """[num_reqs] committed positions before this step."""
    seq_lens: np.ndarray
    """[num_reqs] committed positions plus this step's tokens."""
    positions: np.ndarray
    """[num_tokens] target position of every token."""
    slot_mapping: np.ndarray
    """[num_tokens] draft KV slot of every token."""
    block_table_updates: dict[int, BlockTableUpdate]
    """Block table changes per request slot, to apply before the step."""
    num_sampled: np.ndarray
    num_rejected: np.ndarray
    last_sampled: np.ndarray
    next_prefill_tokens: np.ndarray
    """[num_reqs, num_prefill_lookahead]"""

    @property
    def num_reqs(self) -> int:
        return len(self.rows)

    @property
    def num_tokens(self) -> int:
        return int(self.query_start_loc[-1])


@dataclass
class _DraftSequence:
    req_index: int
    num_computed: int = 0


class DraftStepPlanner:
    """Request slots, draft KV blocks, and committed lengths per sequence.

    ``max_num_reqs`` bounds open sequences, matching the per-request buffers
    a speculator allocates. Each sequence keeps KV blocks for its committed
    length, the next query, and the positions used by additional draft forwards.
    """

    def __init__(
        self,
        block_pool: DraftBlockPool,
        *,
        max_num_reqs: int,
        max_model_len: int,
        num_speculative_tokens: int,
        num_prefill_lookahead: int = 1,
    ) -> None:
        self.block_pool = block_pool
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.num_speculative_tokens = num_speculative_tokens
        self.num_prefill_lookahead = num_prefill_lookahead
        self._free_slots = list(range(max_num_reqs - 1, -1, -1))
        self._num_free_slots = max_num_reqs
        self._sequences: dict[SequenceKey, _DraftSequence] = {}

    def __contains__(self, key: SequenceKey) -> bool:
        return key in self._sequences

    @property
    def num_open(self) -> int:
        return len(self._sequences)

    def req_index(self, key: SequenceKey) -> int:
        return self._sequences[key].req_index

    def num_computed(self, key: SequenceKey) -> int:
        return self._sequences[key].num_computed

    def open(self, key: SequenceKey) -> bool:
        """Assign a request slot; False when every slot is in use."""
        if key in self._sequences:
            raise ValueError(f"sequence {key} is already open")
        if not self._num_free_slots:
            return False
        remaining = self._num_free_slots - 1
        self._sequences[key] = _DraftSequence(self._free_slots[remaining])
        self._num_free_slots = remaining
        return True

    def close(self, key: SequenceKey) -> None:
        sequence = self._sequences.get(key)
        if sequence is not None:
            returned = self._num_free_slots + 1
            self._free_slots[self._num_free_slots] = sequence.req_index
            self.block_pool.release(sequence.req_index)
            del self._sequences[key]
            self._num_free_slots = returned

    def plan(
        self,
        rows: Sequence[DraftStepRow],
        *,
        num_speculative_tokens: int | None = None,
    ) -> tuple[DraftStepPlan, dict[int, tuple[SpeculatorStatusCode, str]]]:
        """Admit rows into one step and reserve their KV.

        Returns the plan for the admitted rows and the failure of every row
        that was left out, keyed by its index in ``rows``. Malformed rows
        raise ValueError: they indicate a bug in the caller, not a
        per-sequence condition.
        """
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_tokens
        if not 0 <= num_speculative_tokens <= self.num_speculative_tokens:
            raise ValueError("round draft budget exceeds the configured maximum")
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("a draft step cannot cover a sequence twice")
        admitted: list[DraftStepRow] = []
        updates: dict[int, BlockTableUpdate] = {}
        failures: dict[int, tuple[SpeculatorStatusCode, str]] = {}
        for i, row in enumerate(rows):
            self._validate(row)
            sequence = self._sequences.get(row.key)
            if sequence is None:
                failures[i] = (SpeculatorStatusCode.SEQUENCE_DESYNC, "not open")
                continue
            seq_len = sequence.num_computed + row.num_tokens
            if seq_len > self.max_model_len:
                failures[i] = (
                    SpeculatorStatusCode.OUT_OF_MEMORY,
                    f"sequence would reach {seq_len} positions, beyond "
                    f"max_model_len={self.max_model_len}",
                )
                continue
            start = len(self.block_pool.blocks(sequence.req_index))
            draft_extent = min(
                self.max_model_len,
                seq_len - row.num_rejected + max(num_speculative_tokens - 1, 0),
            )
            try:
                blocks = self.block_pool.ensure(
                    sequence.req_index, max(seq_len, draft_extent)
                )
            except DraftKVExhausted as e:
                failures[i] = (SpeculatorStatusCode.OUT_OF_MEMORY, str(e))
                continue
            if blocks:
                updates[sequence.req_index] = BlockTableUpdate(start, blocks)
            admitted.append(row)
        return self._build(admitted, updates), failures

    def commit(self, plan: DraftStepPlan) -> None:
        """Advance every planned row by its accepted tokens."""
        for row in plan.rows:
            self._sequences[row.key].num_computed += row.num_tokens - row.num_rejected

    def _validate(self, row: DraftStepRow) -> None:
        if row.num_tokens <= 0:
            raise ValueError(f"row for {row.key} covers no tokens")
        if not 0 <= row.num_rejected < row.num_tokens:
            raise ValueError(
                f"row for {row.key} rejects {row.num_rejected} of "
                f"{row.num_tokens} tokens"
            )
        if row.num_sampled < 0:
            raise ValueError(f"row for {row.key} samples {row.num_sampled} tokens")
        if row.num_sampled == 0:
            if row.num_rejected:
                raise ValueError(f"prompt chunk for {row.key} rejects tokens")
            if len(row.next_prefill_tokens) != self.num_prefill_lookahead:
                raise ValueError(
                    f"prompt chunk for {row.key} carries "
                    f"{len(row.next_prefill_tokens)} lookahead tokens, expected "
                    f"{self.num_prefill_lookahead}"
                )

    def _build(
        self, rows: list[DraftStepRow], updates: dict[int, BlockTableUpdate]
    ) -> DraftStepPlan:
        req_indices = np.array(
            [self._sequences[row.key].req_index for row in rows], dtype=np.int32
        )
        num_computed = np.array(
            [self._sequences[row.key].num_computed for row in rows], dtype=np.int32
        )
        num_scheduled = np.array([row.num_tokens for row in rows], dtype=np.int32)
        query_start_loc = np.zeros(len(rows) + 1, dtype=np.int32)
        np.cumsum(num_scheduled, out=query_start_loc[1:])
        positions = np.concatenate(
            [
                np.arange(start, start + count, dtype=np.int64)
                for start, count in zip(num_computed, num_scheduled)
            ]
            or [np.empty(0, dtype=np.int64)]
        )
        slot_mapping = np.concatenate(
            [
                self.block_pool.slot_mapping(
                    int(req_index),
                    positions[query_start_loc[i] : query_start_loc[i + 1]],
                )
                for i, req_index in enumerate(req_indices)
            ]
            or [np.empty(0, dtype=np.int64)]
        )
        lookahead = np.zeros((len(rows), self.num_prefill_lookahead), dtype=np.int64)
        for i, row in enumerate(rows):
            if row.num_sampled == 0:
                lookahead[i] = row.next_prefill_tokens
        return DraftStepPlan(
            rows=tuple(rows),
            req_indices=req_indices,
            num_scheduled_tokens=num_scheduled,
            query_start_loc=query_start_loc,
            num_computed_tokens=num_computed,
            seq_lens=num_computed + num_scheduled,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table_updates=updates,
            num_sampled=np.array([row.num_sampled for row in rows], dtype=np.int32),
            num_rejected=np.array([row.num_rejected for row in rows], dtype=np.int32),
            last_sampled=np.array([row.last_sampled for row in rows], dtype=np.int64),
            next_prefill_tokens=lookahead,
        )
