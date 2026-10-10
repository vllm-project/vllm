# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU transport for the V2 worker's DP batch metadata."""

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm import envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def recursive_doubling_sum_(tensor, group, ranks, rank_in_group, scratch=None):
    """Sum in place with serialized CPU exchanges on a power-of-two group.

    All members must call in the same order. Tags starting at 27100 are
    reserved for this operation on the supplied group. Each stage waits for
    both send and receive before modifying or reusing either buffer.
    """
    size = len(ranks)
    assert size > 1 and size & (size - 1) == 0
    assert tensor.device.type == "cpu" and tensor.is_contiguous()
    assert tensor.dtype == torch.int32
    if scratch is None:
        scratch = torch.empty_like(tensor)
    assert scratch.data_ptr() != tensor.data_ptr()
    mask = 1
    stage = 0
    while mask < size:
        peer = ranks[rank_in_group ^ mask]
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.irecv, scratch, peer, group=group, tag=27100 + stage),
                dist.P2POp(dist.isend, tensor, peer, group=group, tag=27100 + stage),
            ]
        )
        for work in works:
            work.wait()
        tensor.add_(scratch)
        mask <<= 1
        stage += 1
    return tensor


@lru_cache(maxsize=1)
def _metadata_scratch(shape: tuple[int, ...]) -> torch.Tensor:
    # Calls in each worker are synchronous and serialized.
    return torch.empty(shape, dtype=torch.int32, device="cpu")


def all_reduce_dp_metadata(tensor: torch.Tensor, dp_group: "GroupCoordinator") -> None:
    """Sum CPU metadata in place, optionally using recursive doubling."""
    mode = envs.VLLM_DP_METADATA_TRANSPORT
    if mode == "gloo":
        dist.all_reduce(tensor, group=dp_group.cpu_group)
    else:
        recursive_doubling_sum_(
            tensor,
            dp_group.cpu_group,
            dp_group.ranks,
            dp_group.rank_in_group,
            _metadata_scratch(tuple(tensor.shape)),
        )
