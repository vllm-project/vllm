# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    pp = get_pp_group()
    if not pp.is_last_rank:
        return

    assert sampled_token_ids.dtype == torch.int64
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(),
        src=pp.last_rank,
        group=pp.device_group,
    )
    torch.distributed.broadcast(
        num_sampled.contiguous(),
        src=pp.last_rank,
        group=pp.device_group,
    )
    torch.distributed.broadcast(
        num_rejected.contiguous(),
        src=pp.last_rank,
        group=pp.device_group,
    )


def pp_receive(
    num_reqs: int,
    max_sample_len: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pp = get_pp_group()
    if pp.is_last_rank:
        return None

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    torch.distributed.broadcast(
        sampled_tokens,
        src=pp.last_rank,
        group=pp.device_group,
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(
        num_sampled,
        src=pp.last_rank,
        group=pp.device_group,
    )
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(
        num_rejected,
        src=pp.last_rank,
        group=pp.device_group,
    )
    return sampled_tokens, num_sampled, num_rejected
