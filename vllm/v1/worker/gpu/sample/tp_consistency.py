# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep sampling identical across tensor-parallel ranks.

Every TP rank samples on the same all-gathered logits and feeds its own sampled
token into the next step; only TP rank 0's tokens reach the user, and nothing
compares the ranks afterwards. So all TP ranks must pick the same sampler backend
and start from the same random state, or they silently continue on different
tokens. Both checks below run once, when the sampler is built.
"""

from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _tp_cpu_group() -> Any | None:
    """The TP group's CPU (gloo) group, or None when there is nothing to agree on."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import (
        get_tp_group,
        model_parallel_is_initialized,
    )

    if not (dist.is_initialized() and model_parallel_is_initialized()):
        return None
    group = get_tp_group()
    return group.cpu_group if group.world_size > 1 else None


def all_tp_ranks_agree(local: bool) -> bool:
    """True only if `local` is True on every TP rank (a collective on the TP group)."""
    group = _tp_cpu_group()
    if group is None:
        return local
    import torch.distributed as dist

    flag = torch.tensor([int(local)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def check_sampling_config(make_config: Callable[[], dict[str, Any]]) -> None:
    """Raise if TP ranks were configured to sample differently (a collective).

    `make_config` is only called when there are TP ranks to compare."""
    group = _tp_cpu_group()
    if group is None:
        return
    import torch.distributed as dist

    config = make_config()

    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, config, group=group)
    keys = sorted({k for c in gathered if c is not None for k in c})
    differing = {
        k: [None if c is None else c.get(k) for c in gathered]
        for k in keys
        if len({repr(None if c is None else c.get(k)) for c in gathered}) > 1
    }
    if differing:
        detail = "; ".join(f"{k} per TP rank = {v}" for k, v in differing.items())
        raise RuntimeError(
            f"Tensor-parallel ranks are configured to sample differently ({detail}). "
            "Each rank samples its own tokens and feeds them into the next step, so "
            "they would silently diverge from the tokens returned to the user. Start "
            "every node with the same sampling arguments (for example --seed)."
        )
