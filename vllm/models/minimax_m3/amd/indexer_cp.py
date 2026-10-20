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
    """Allocate ``nbytes`` on every rank and return all their addresses.

    Comes back as a ``[world]`` int64 CPU tensor, entry ``r`` being rank
    ``r``'s buffer as addressed from here -- this rank's own at its own index,
    an IPC mapping everywhere else. That is the form the kernel's peer table
    takes, and it is read on the host at launch, so a replay of a captured
    graph needs the addresses to still be valid: nothing frees them.

    ``create_shared_buffer`` is the custom all-reduce's own allocator, reused
    rather than reimplemented. It wants the same three things this does: a
    plain device address outside torch's caching allocator, since under
    expandable segments a torch pointer lives in a ``hipMallocAsync`` pool and
    cannot be exported over IPC at all; uncached memory, since the merge spins
    on candidates a peer wrote and has to read them past this device's caches,
    which its C++ gets from ``hipExtMallocWithFlags`` on ROCm; and the handle
    exchanged over the gloo group, because the device group is NCCL.

    It also zeroes the buffer, which is load-bearing and happens only here. A
    candidate carries the generation that wrote it and no generation is ever
    0, so zero is what the merge reads as "has not arrived". Re-zeroing later
    would not just be unnecessary, it would race a peer that had not yet read
    the previous round.
    """
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    nbytes = (nbytes + _HIP_PAGE - 1) // _HIP_PAGE * _HIP_PAGE
    peers = CustomAllreduce.create_shared_buffer(nbytes, group=group.cpu_group)
    return torch.tensor(peers, dtype=torch.int64, device="cpu")


@dataclass(frozen=True)
class IndexerCpPeers:
    """The buffers the CP selector exchanges through.

    ``cand_ptrs`` is a ``[world]`` int64 CPU tensor of device addresses, which
    is what ``pa_sparse_block_topk_cp`` takes: it builds the kernel's peer
    table on the host so the pointers arrive in registers rather than behind a
    load. ``cp_gen`` is this rank's own, not mapped to anyone -- a candidate
    carries the generation that wrote it, so the only shared state is the
    candidates themselves.
    """

    cand_ptrs: torch.Tensor
    cp_gen: torch.Tensor


_PEERS: dict[tuple[int, int, int, int], IndexerCpPeers] = {}


def get_indexer_cp_peers(owned_heads: int, max_rows: int, topk: int) -> IndexerCpPeers:
    """The buffers the CP selector exchanges through, allocated once.

    Persistent and shared by every layer rather than allocated per launch. The
    addresses are what make that necessary: they are read on the host and baked
    into captured graphs, so a replay has to find the same buffer, and nothing
    is ever freed. Fifty-odd layers each mapping their own would be fifty-odd
    IPC mappings per rank for buffers only one layer at a time is inside.

    What lets one buffer serve the whole stack is the generation in each
    candidate. A layer's writes land in the half of the buffer the previous
    layer is not reading, and carry a tag saying which call produced them, so
    consecutive layers never have to be separated by a barrier.

    Keyed by shape because the buffers are sized to it, and by head count
    because the generation counters are indexed by the launch's head extent --
    two head counts sharing a counter array would be comparing tags across
    different calls. Every rank must reach this with the same key; they are
    allocating against each other.

    ``max_rows`` is the widest decode batch, not the current one, for the same
    capture reason: the size has to be the one every captured batch addresses,
    and a short batch simply leaves the tail unused.
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
        # Not IPC and not zeroed again after this: the counter is read only by
        # the rank that owns it, and a stale value is exactly what the next
        # call distinguishes itself from.
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
