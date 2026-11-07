# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types of the speculative round contract (see ``round_contract.md``).

Shapes use ``R = max_num_reqs`` (by batch row, the order of ``idx_mapping``) and
``C = num_speculative_tokens + 1`` (rows per request, the root included).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


def check_rows(
    name: str, t: torch.Tensor, ndim: int, dtype: torch.dtype | None = torch.int32
) -> None:
    """Raise unless ``t`` is an ``ndim``-D tensor of ``dtype`` (any dtype for
    None) with unit stride along its last dimension, the layout the round's
    kernels index."""
    if (
        (dtype is not None and t.dtype != dtype)
        or t.dim() != ndim
        or (t.shape[-1] > 1 and t.stride(-1) != 1)
    ):
        raise ValueError(
            f"{name} must be a {ndim}-D {dtype or ''} tensor with unit stride "
            f"along its last dimension; got shape {tuple(t.shape)}, {t.dtype}, "
            f"strides {t.stride()}"
        )


@dataclass
class VerifyPlan:
    """The rows the target verifies this step, per scheduled request.

    Row 0 is the root; ``parent[r, i]`` is in ``[0, i)``, so rows are ordered
    parent first and a chain has ``parent[r, i] == i - 1``. Entries past a
    request's rows are ignored. Buffers stay valid until the commit returns.
    """

    parent: torch.Tensor
    """``[R, C]`` int32, ``parent[r, 0] == -1``."""
    tokens: torch.Tensor | None = None
    """``[R, C]`` int32 candidate tokens (``tokens[r, 0]`` is ignored: the root token is
    known). None exactly for a chain, whose tokens are in the runner's draft
    buffer."""
    is_chain: bool = False
    """Host flag: every ``parent[r, i] == i - 1`` by construction. The runner
    then keeps chain positions and compacts no KV."""
    rows: np.ndarray | None = None
    """``[num_reqs]`` host int32: the rows each request verifies, at most the
    ``PlanInputs.scheduled_rows`` the scheduler reserved; the drafts past them leave
    the batch and count as rejected. A row with no candidate holds token -1. None
    verifies every scheduled row."""

    def __post_init__(self) -> None:
        check_rows("VerifyPlan.parent", self.parent, 2)
        if self.tokens is not None:
            check_rows("VerifyPlan.tokens", self.tokens, 2)
            if self.tokens.shape != self.parent.shape:
                raise ValueError("a plan's tokens must have the shape of its parents")
        if not self.is_chain and self.tokens is None:
            raise ValueError("a plan that is not a chain must carry tokens")
        if self.is_chain and self.tokens is not None:
            raise ValueError("a chain plan must not carry tokens: they are drafted")


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
    """The plan must be a chain this step (``round_contract.md``, step 2); it may
    still verify fewer rows."""


@dataclass
class AcceptResult:
    """The accepted rows of each request and the committed tokens.

    ``path[r, :path_len[r]]`` lists the accepted rows from the root down
    (``path[r, j] >= j``); ``sampled[r, :path_len[r]]`` are the accepted tokens
    followed by the bonus token.
    """

    path: torch.Tensor
    """``[R, C]`` int32 row indices."""
    path_len: torch.Tensor
    """``[R]`` int32 in ``[1, C]``: the number of committed tokens (0 for a row that
    sampled nothing, e.g. a prefill chunk that does not end its prompt)."""
    sampled: torch.Tensor
    """``[R, C]`` token ids."""


@dataclass(frozen=True)
class TargetTokens:
    """What the target chose at the verify rows of a range of requests, flat in
    plan order (every tensor indexed from the range's first row)."""

    tokens: torch.Tensor
    """``[num_rows]`` int64: the target's token at each row, the argmax or a
    draw from the row's processed logits at its plan position."""
    drafted: torch.Tensor
    """``[num_rows]`` int64: the token each row holds (row 0: the last committed
    token); -1 for a row that must not be accepted (a placeholder, or a draft
    the grammar rejected)."""
    row_start: torch.Tensor
    """``[num_reqs + 1]`` int32: where each request's rows start."""
