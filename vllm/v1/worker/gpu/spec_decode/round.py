# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types of the speculative round contract.

A speculative step is a round: the speculator plans the rows the target verifies
(:class:`VerifyPlan`), the target runs them, the speculator accepts a path through them
(:class:`AcceptResult`) and the runner commits the KV and the recurrent state along that
path. :class:`SpecResources` is what speculation holds, declared once at startup, and
:class:`StepObservation` is the read-only record of each step.

Everything a speculator hands the runner is either a frozen dataclass of host integers
or a device tensor in a buffer the speculator owns and keeps alive for the step. The
runner never reads a device value on the host.

Shapes use ``R = max_num_reqs`` (indexed by batch row, the order of ``idx_mapping``)
and ``C = SpecResources.max_candidates`` (rows per request, the root included).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.capabilities import PlanKind
from vllm.v1.worker.gpu.spec_decode.step_observation import (
    StepObservation as _StepObservation,
)


@dataclass(frozen=True)
class PlanShape:
    """One target-forward shape the runner captures as a CUDA graph.

    ``uniform_count`` rows per request when set (a chain's K + 1, or a fixed tree
    width). None asks for a variable-length decode graph that takes any mix of
    1..max_candidates rows per request.
    """

    uniform_count: int | None


@dataclass(frozen=True)
class SpecResources:
    """What speculation holds.

    The scheduler reserves ``max_candidates`` KV slots per decoding request and the
    runner captures ``capture_shapes``.

    The speculator's config declares them (``SpeculatorConfig.resources``), and the
    KV and recurrent-state pools are sized for that declaration. The engine core then
    lowers them, never raises them, to what the model's layers verify
    (``SpeculatorConfig.adapt_resources``) and hands the result to the scheduler and
    to every worker.
    """

    max_candidates: int
    """Rows per request in a verify forward, the root row included (1 = plain decode).
    The ceiling of every ``VerifyPlan.count``."""
    capture_shapes: tuple[PlanShape, ...]
    max_branching: int = 1
    """1 for chains. Above 1 the attention backend and the linear-attention layers must
    take a parent per row."""
    custom_mask: bool = False
    """Plans may carry ``VerifyPlan.mask``: the attention layers must take a
    visibility mask per row. Needs ``max_branching > 1``."""

    @property
    def plan_kind(self) -> PlanKind:
        """The plans the speculator verifies: masks when it declares
        ``custom_mask``, trees when it branches, else chains."""
        if self.custom_mask:
            return "mask"
        return "tree" if self.max_branching > 1 else "chain"

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}.")
        bad = [
            shape.uniform_count
            for shape in self.capture_shapes
            if shape.uniform_count is not None
            and not 1 <= shape.uniform_count <= self.max_candidates
        ]
        if bad:
            raise ValueError(
                f"capture_shapes uniform counts {bad} are outside "
                f"[1, max_candidates={self.max_candidates}]."
            )
        if self.max_branching < 1:
            raise ValueError(
                f"max_branching ({self.max_branching}) must be >= 1: 1 is a chain."
            )
        if self.custom_mask and self.max_branching == 1:
            raise ValueError(
                "custom_mask needs max_branching > 1: a plan with a mask is laid "
                "out as a tree"
            )

    @property
    def uniform_counts(self) -> list[int]:
        """The uniform rows per request of the listed shapes, ascending."""
        return sorted(
            {
                s.uniform_count
                for s in self.capture_shapes
                if s.uniform_count is not None
            }
        )

    @property
    def varlen(self) -> bool:
        """Whether a variable-length decode graph is listed."""
        return any(s.uniform_count is None for s in self.capture_shapes)


@dataclass(frozen=True)
class BudgetView:
    """What the scheduler knows when it asks for the step's admission limit
    (:meth:`SpeculatorConfig.admit_limit`) and candidate budget
    (:meth:`SpeculatorConfig.candidate_budget`). Host integers. For the limit
    they are read once the running requests are scheduled, before any waiting
    one is admitted; for the budget, once all of this step's requests are
    scheduled and their blocks allocated."""

    num_scheduled_reqs: int
    num_running: int
    num_waiting: int
    num_free_blocks: int
    max_num_seqs: int
    block_size: int
    """Tokens per block of the unified pool (the scheduler's block size)."""


@dataclass
class VerifyPlan:
    """The rows the target verifies this step, per scheduled request, as device tensors.

    Row 0 of a request is the root: its last committed token, whose logits sample the
    bonus token. Rows ``1..count-1`` are candidates; ``parent[r, i]`` is the row that
    candidate ``i`` continues, in ``[0, i)``, so rows are ordered parent first and any
    prefix of them is a plan too. A chain has ``parent[r, i] == i - 1``. Entries past
    ``count[r]`` are ignored.

    The runner derives from the plan the position of each row (``num_computed_tokens``
    plus its depth, or plus ``positions``), the rows it sees (its ancestors, or
    ``mask``), and its KV slot: every row writes its KV at the
    chain slot of its row index, and the commit compacts the accepted path afterwards.

    Buffers belong to whoever built the plan and stay valid until the commit returns,
    so CUDA graph replays read stable addresses.
    """

    parent: torch.Tensor
    """``[R, C]`` int32, ``parent[r, 0] == -1``."""
    count: torch.Tensor | None
    """``[R]`` int32 in ``[1, C]``; 1 is a plain decode. None only for the chain plan
    the runner builds itself, whose counts are the scheduled drafts plus one."""
    tokens: torch.Tensor | None = None
    """``[R, C]`` int32 candidate tokens (``tokens[r, 0]`` is ignored: the root token is
    known). None only for the runner's chain plan, whose tokens are in its draft
    buffer."""
    draft_probs: torch.Tensor | None = None
    """``[R, C, vocab]`` float32 for probabilistic verification, else None."""
    is_chain: bool = False
    """Host flag set by the builder when every ``parent[r, i] == i - 1`` by
    construction. The runner then runs the chain-only path: positions stay
    ``num_computed_tokens + i``, no ancestor sets, no KV compaction. Never derived
    from the device tensors."""
    positions: torch.Tensor | None = None
    """``[R, C]`` int32 position of each row as an offset from
    ``num_computed_tokens``, replacing the row's depth; None places each row at its
    depth. Positions do not enter a recurrence, so any model takes them. A row's KV
    slot stays its row index."""
    mask: torch.Tensor | None = None
    """``[R, C, C]`` bool: ``mask[r, i, j]`` is whether row ``i`` sees row ``j``;
    None sees the row's ancestors. ``mask[r, i, i]`` is taken as True and entries
    with ``j > i`` are ignored (a row never sees a later row); the committed prefix
    is always visible. Needs resources with ``custom_mask`` (plan kind
    ``"mask"``). ``parent`` still gives the paths the commit accepts along, and the
    recurrent layers of a hybrid model read ``parent``, not the mask."""

    def __post_init__(self) -> None:
        if not self.is_chain and (self.count is None or self.tokens is None):
            raise ValueError("a plan that is not a chain must carry count and tokens")
        if self.is_chain and (self.positions is not None or self.mask is not None):
            raise ValueError(
                "a chain plan has positions num_computed_tokens + i and causal "
                "attention; a plan with positions or a mask is not a chain"
            )
        rows = self.parent.shape
        if self.positions is not None and tuple(self.positions.shape) != tuple(rows):
            raise ValueError(
                f"positions has shape {tuple(self.positions.shape)}, parent "
                f"{tuple(rows)}"
            )
        if self.mask is not None and tuple(self.mask.shape) != (*rows, rows[1]):
            raise ValueError(
                f"mask has shape {tuple(self.mask.shape)}, expected "
                f"{(*rows, rows[1])} for parent {tuple(rows)}"
            )


@dataclass(frozen=True)
class PlanInputs:
    """What the runner knows about a batch before its inputs are built, for
    :meth:`BaseSpeculator.plan`. Host values, except ``idx_mapping``."""

    req_ids: Sequence[str]
    """The batch's requests in batch-row order. Stable across preemption, unlike
    the state slots."""
    idx_mapping: torch.Tensor
    """``[num_reqs]`` int64 device tensor: the request state slot of each batch row."""
    idx_mapping_np: np.ndarray
    """``[num_reqs]`` host copy of ``idx_mapping``."""
    scheduled_rows: np.ndarray
    """``[num_reqs]`` int32: rows the scheduler reserved per request, 1 + its drafts."""
    chain_only: bool
    """The runner can only verify chains this step: a prefill is in the batch, a
    logits processor reads each row's prefix (penalties, bad words, a thinking
    budget, a custom processor), structured output, or a parallelism mode that does
    not lay out other plans (pipeline, context parallel). The plan must then
    be a chain; it may still verify fewer rows."""


@dataclass(frozen=True)
class PlanBound:
    """Host integers the runner needs before the device plan exists.

    ``rows[r]`` is how many rows request ``r`` verifies, at most the
    ``scheduled_rows[r]`` the scheduler reserved (1 + its drafts); the drafts past
    it leave the batch and count as rejected. None verifies every scheduled row.
    """

    rows: np.ndarray | None = None


@dataclass
class VerifyOutput:
    """What the target computed for the rows of the requests ``accept`` verifies,
    in plan order. A large batch is verified in request ranges; every tensor here
    and the plan ``accept`` gets cover the same range, indexed from 0."""

    logits: torch.Tensor
    """``[num_rows, vocab]`` raw logits after the LM head (and the grammar bitmask)."""
    row_start: torch.Tensor
    """``[num_reqs + 1]`` int32 cumulative row offsets of each request."""
    idx_mapping: torch.Tensor
    """``[num_reqs]`` request state slot of each request."""
    idx_mapping_np: np.ndarray
    expanded_idx_mapping: torch.Tensor
    """``[num_rows]`` request state slot of each row."""
    expanded_local_pos: torch.Tensor
    """``[num_rows]`` row index within its request (its plan row)."""
    positions: torch.Tensor
    """``[num_rows]`` int64 plan position of each row: the request's committed
    tokens plus the row's depth. Sampling noise and position-dependent logits
    processing use it."""
    drafted: torch.Tensor
    """``[num_rows]`` int64 token each row holds (row 0: the last committed token); -1
    for a row that must not be accepted (a placeholder or a grammar-rejected draft)."""
    draft_tokens: torch.Tensor
    """``[num_rows]`` the rows' real token ids, for logits processing."""
    seq_lens_upper_bound_np: np.ndarray
    """``[num_reqs]`` host upper bound of each request's sequence length."""


@dataclass
class AcceptResult:
    """Which rows of each request the target confirmed, and the token that follows.

    ``path[r, :path_len[r]]`` lists the accepted rows from the root (row 0) down, each
    a child of the previous one, so ``path[r, j] >= j``. ``sampled[r, :path_len[r]]``
    are the committed tokens: the accepted candidates' tokens followed by the bonus
    token sampled from the logits of the last accepted row. The runner commits the KV
    and the recurrent state along ``path``; every other row is rejected. Whether the
    commit moves anything is the plan's ``is_chain``, not a property of the result.
    """

    path: torch.Tensor
    """``[R, C]`` int32 row indices."""
    path_len: torch.Tensor
    """``[R]`` int32 in ``[1, C]``: the number of committed tokens (0 for a row that
    sampled nothing, e.g. a prefill chunk that does not end its prompt)."""
    sampled: torch.Tensor
    """``[R, C]`` token ids."""
    processed_logits: torch.Tensor | None = None
    """``[num_rows, vocab]`` the logits after the sampling parameters, in row order,
    when ``accept`` computed them: the runner takes processed logprobs and the
    sampling mask of each committed token from its path row. None: the raw logits."""


# Defined with observe_step (step_observation.py); re-exported as part of the
# round's vocabulary.
StepObservation = _StepObservation
