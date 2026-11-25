# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep sampling identical across tensor-parallel ranks.

Every TP rank samples on the same all-gathered logits and feeds its own sampled
token into the next step; only TP rank 0's tokens reach the user, and nothing
compares the ranks afterwards. So all TP ranks must pick the same sampler backend
and start from the same random state, or they silently continue on different
tokens. The check below runs once, when the sampler is built.
"""

from typing import Any


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


def agree_on_sampling(
    config: dict[str, Any], supported: dict[str, bool]
) -> dict[str, bool]:
    """Compare the sampling config across TP ranks and agree on fused samplers.

    One collective on the TP group: raise if `config` differs between ranks, and
    return each entry of `supported` as True only if every rank supports it."""
    group = _tp_cpu_group()
    if group is None:
        return dict(supported)
    import torch.distributed as dist

    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, (config, supported), group=group)
    configs = [c for c, _ in gathered]
    keys = sorted({k for c in configs for k in c})
    differing = {
        k: [c.get(k) for c in configs]
        for k in keys
        if len({repr(c.get(k)) for c in configs}) > 1
    }
    if differing:
        detail = "; ".join(f"{k} per TP rank = {v}" for k, v in differing.items())
        raise RuntimeError(
            f"Tensor-parallel ranks are configured to sample differently ({detail}). "
            "Each rank samples its own tokens and feeds them into the next step, so "
            "they would silently diverge from the tokens returned to the user. Start "
            "every node with the same sampling arguments (for example --seed)."
        )
    return {k: all(s.get(k, False) for _, s in gathered) for k in supported}
