# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout a :class:`VerifyPlan` gives the target's verify forward.

Each request verifies the last ``count[b]`` rows of its query; row 0 of those is the
root. The runner computes the batch's positions and slot mappings as for a chain
(``num_computed_tokens + i``).
"""

import torch


def chain_parents(
    max_num_reqs: int, max_rows: int, device: torch.device
) -> torch.Tensor:
    """The ``[R, C]`` parent table of a chain: ``parent[r, i] == i - 1``."""
    rows = torch.arange(-1, max_rows - 1, dtype=torch.int32, device=device)
    return rows.expand(max_num_reqs, max_rows).contiguous()
