# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types of the speculative round contract.

A speculative step is a round: the speculator plans the rows the target verifies
(:class:`VerifyPlan`), the target runs them, the runner accepts a path through them
(:class:`AcceptResult`) and commits the KV and the recurrent state along that
path. What speculation holds is :class:`vllm.v1.spec_decode.resources.SpecResources`.

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
