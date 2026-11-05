# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types of the speculative round contract.

A speculative step is a round: the speculator plans the rows the target verifies
(:class:`VerifyPlan`), the target runs them, the speculator accepts a path through them
(:class:`AcceptResult`) and the runner commits the KV and the recurrent state along that
path. :class:`SpecResources` is what speculation holds, declared once at startup.

Everything a speculator hands the runner is either a frozen dataclass of host integers
or a device tensor in a buffer the speculator owns and keeps alive for the step. The
runner never reads a device value on the host.

Shapes use ``R = max_num_reqs`` (indexed by batch row, the order of ``idx_mapping``)
and ``C = num_speculative_tokens + 1`` (rows per request, the root included).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.capabilities import PlanKind


@dataclass(frozen=True)
class SpecResources:
    """What speculation holds.

    The runner captures a decode CUDA graph for each entry of ``capture_rows``,
    and no request verifies more rows than the largest (``max_rows``).

    The speculator's config declares them (``SpeculatorConfig.resources``), and the
    KV and recurrent-state pools are sized for ``num_speculative_tokens``. The engine
    core then lowers them, never raises them, to what the model's layers verify
    (``SpeculatorConfig.adapt_resources``) and hands the result to the scheduler and
    to every worker.
    """

    capture_rows: tuple[int, ...]
    """Rows per request, the root included, of each decode graph (1 = plain
    decode); at most ``num_speculative_tokens + 1``."""
    max_branching: int = 1
    """1 for chains. Above 1 the attention backend and the linear-attention layers must
    take a parent per row."""

    @property
    def plan_kind(self) -> PlanKind:
        """The plans the speculator verifies: trees when it branches."""
        return "tree" if self.max_branching > 1 else "chain"

    @property
    def max_rows(self) -> int:
        """The most rows a request verifies, the root included."""
        return max(self.capture_rows)

    def __post_init__(self) -> None:
        bad = [n for n in self.capture_rows if not isinstance(n, int) or n < 1]
        if not self.capture_rows or bad:
            raise ValueError(
                f"capture_rows must be positive ints, at least one; got "
                f"{self.capture_rows}."
            )
        if self.max_branching < 1:
            raise ValueError(
                f"max_branching ({self.max_branching}) must be >= 1: 1 is a chain."
            )


@dataclass(frozen=True)
class BudgetView:
    """What the scheduler knows when it asks for the step's candidate budget
    (:meth:`SpeculatorConfig.candidate_budget`). Host integers, read once all
    of this step's requests are scheduled and their blocks allocated."""

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
    bonus token. Rows ``1..rows[r]-1`` are candidates; ``parent[r, i]`` is the row
    that candidate ``i`` continues, in ``[0, i)``, so rows are ordered parent first and
    any prefix of them is a plan too. A chain has ``parent[r, i] == i - 1``. Entries
    past a request's rows are ignored.

    The runner derives from the plan the position of each row (``num_computed_tokens``
    plus its depth), its ancestor set, and its KV slot: every row writes its KV
    at the chain slot of its row index, and the commit compacts the accepted path
    afterwards.

    Buffers belong to whoever built the plan and stay valid until the commit returns,
    so CUDA graph replays read stable addresses.
    """

    parent: torch.Tensor
    """``[R, C]`` int32, ``parent[r, 0] == -1``."""
    tokens: torch.Tensor | None = None
    """``[R, C]`` int32 candidate tokens (``tokens[r, 0]`` is ignored: the root token is
    known). None only for the runner's chain plan, whose tokens are in its draft
    buffer."""
    is_chain: bool = False
    """Host flag set by the builder when every ``parent[r, i] == i - 1`` by
    construction. The runner then runs the chain-only path: positions stay
    ``num_computed_tokens + i``, no ancestor sets, no KV compaction. Never derived
    from the device tensors."""
    rows: np.ndarray | None = None
    """``[num_reqs]`` host int32: the rows each request verifies, at most the
    ``PlanInputs.scheduled_rows`` the scheduler reserved; the drafts past them leave
    the batch and count as rejected. A row with no candidate holds token -1. None
    verifies every scheduled row."""

    def __post_init__(self) -> None:
        if not self.is_chain and self.tokens is None:
            raise ValueError("a plan that is not a chain must carry tokens")


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
