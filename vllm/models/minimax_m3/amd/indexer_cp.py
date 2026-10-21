# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer-only context parallelism for MiniMax M3 (AMD MSA indexer).

Every rank projects all H index heads and scores 1/P of the context blocks, then
one exchange gives it each shard's candidates for the heads it owns, cutting the
replicated index-cache read to 1/P. Shards cut to the full k so selection stays
exact; candidates carry a generation tag, so all ranks must call this in lockstep.
"""

from dataclasses import dataclass

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_indexer_cp_group():
    """The group the candidate exchange runs over."""
    return get_tp_group()


# Granularity an uncached allocation under 2MB has to be a whole number of
# for the IPC export to succeed. At or above 2MB it is backed coarse-grained
# and exports regardless, but rounding up is free.
_HIP_PAGE = 4096


def _ipc_alloc(nbytes: int, group) -> torch.Tensor:
    """Allocate a zeroed, IPC-shared ``nbytes`` on every rank.

    Returns a ``[world]`` int64 CPU tensor of their device addresses, never freed.
    """
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    nbytes = (nbytes + _HIP_PAGE - 1) // _HIP_PAGE * _HIP_PAGE
    peers = CustomAllreduce.create_shared_buffer(nbytes, group=group.cpu_group)
    return torch.tensor(peers, dtype=torch.int64, device="cpu")


@dataclass(frozen=True)
class IndexerCpPeers:
    """The buffers the CP selector exchanges through.

    ``cand_ptrs`` is every rank's candidate buffer; ``cp_gen`` is this rank's own.
    """

    cand_ptrs: torch.Tensor
    cp_gen: torch.Tensor


_PEERS: dict[tuple[int, int, int, int], IndexerCpPeers] = {}


def get_indexer_cp_peers(owned_heads: int, max_rows: int, topk: int) -> IndexerCpPeers:
    """The buffers the CP selector exchanges through, allocated once per key.

    Cached and shared by every layer, sized to ``max_rows`` so captured graphs
    always address the same buffer. Every rank must reach this with the same key.
    """
    from aiter.ops.msa_block_select import CP_MAX_BLOCKS, topk_cp_candidate_numel

    group = get_indexer_cp_group()
    key = (group.world_size, owned_heads, max_rows, topk)
    peers = _PEERS.get(key)
    if peers is not None:
        return peers

    numel = topk_cp_candidate_numel(group.world_size, owned_heads, max_rows, topk)
    peers = IndexerCpPeers(
        cand_ptrs=_ipc_alloc(numel * 8, group),
        cp_gen=torch.zeros(CP_MAX_BLOCKS, dtype=torch.int32, device="cuda"),
    )
    _PEERS[key] = peers
    logger.info_once(
        "MiniMax M3 indexer CP: %d KiB candidate buffer per rank mapped over "
        "%d ranks [owned_heads=%d, rows=%d, topk=%d, generations=2]",
        numel * 8 // 1024,
        group.world_size,
        owned_heads,
        max_rows,
        topk,
    )
    return peers
