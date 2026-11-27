# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-head attention sinks for HY4 ShardQ prefill."""

import torch

from vllm.distributed import get_tp_group


def gather_full_sinks_(sinks: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """All-gather this rank's sinks into `out` across TP.

    `out` is filled in place, so captured graphs read reloaded sinks. It is
    allocated on the first call, when `out` is None.
    """
    tp_group = get_tp_group()
    full = tp_group.all_gather(sinks.to(tp_group.device), dim=0)
    if out is None:
        out = torch.empty_like(full)
    out.copy_(full)
    return out
