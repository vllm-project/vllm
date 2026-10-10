# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer-only context parallelism for MiniMax M3 (AMD MSA indexer).

Inverts the tensor-parallel split for the lightning indexer. Normally a rank
owns ``H/P`` index heads and scores every 128-token block of the context;
under CP it projects all ``H`` heads and scores ``1/P`` of the blocks, then
one exchange hands each rank every shard's candidates for the heads it owns.
Only indexer *work* is sharded. Both caches, the sparse attend, the block
tables and the MoE are untouched, which is what separates this from
``decode_context_parallel_size`` -- that shards the KV cache itself and is
refused alongside this.

What makes it worth doing is that the index cache is replicated, so every
rank re-reads the whole of it every step, and that read is the only part of
the indexer that grows with context. Sharding the block axis turns it into
``1/P`` of the reads: measured at TP4, scoring a 128K context goes from 91us
to 34us at batch 32 and from 364us to 97us at batch 128.

Two things change outside this module. The model projects ``index_q``
replicated rather than split (``replicate_index_q`` in the qkv layer), and
the indexer's head count becomes the model's total rather than this rank's.
Both are decided from one gate answer before any layer is built, which is why
the gate is a branch of ``msa_indexer_unsupported_reason`` and not something
the impl discovers later. For this model it allows ``P <= 4``: there are 4
index heads to go round, and 4 x top-16 is exactly the 64 candidates the
merge holds.

Blocks go round-robin, ``owner(b) = b % P``, with ``global = local * P +
rank``. Contiguous shards would be cheaper to index but worse balanced --
under causal masking rank 0 would get always-full blocks and the last rank
mostly-empty tails, while round-robin keeps the valid counts even at every
sequence length.

Selection stays exact rather than approximate. Each rank cuts its shard to
the full ``k``, never ``k/P``, because the global winners may all live in one
shard; the merge then takes the top-k of the ``P * k`` that arrive. A block
in the global top-k necessarily won its own shard's, so re-selecting over
what arrived reproduces what a rank scoring everything would have picked.

Nominate, exchange and merge are one kernel, ``pa_sparse_block_topk_cp``.
Candidates travel as the packed ``(score, global block id)`` uint64 sort keys
the single-GPU selector already uses to merge across chunks, so nothing on
the wire says which rank a candidate came from and the payload is ``P * k *
8`` bytes per token whatever the context length. They are written straight
into the peer that owns those heads over the IPC mapping set up here, not
all-gathered: at this size a collective costs the same 13us carrying 4KB as
256KB, so what it would spend is launch and synchronisation, and shrinking
the payload never helped.

The buffer is ``[2, P, owned_heads, rows, k]`` int64 on every rank. The shard
axis is the one peers fill, so each writes a disjoint slice and the merge
reads its own buffer with nothing to gather. The leading axis is generation
parity: a call writes the half the previous call is not reading, which is
what lets consecutive layers run back to back without a barrier between them.
Rows are strided by the buffer's own ``cand_rows`` and never by the live
batch, or a short batch would address a different slot than the capture did.

There is no barrier inside the kernel either. A merge has to be told its
peers' nominations landed, and the cheapest way to be told is for the payload
to say so: the id half of each candidate carries the generation that wrote it
in its top 8 bits, and a merging lane re-reads the slot it wants until the
tag matches the generation it is on. The arrival flag is the data, so there
is no flag to clear and no second round trip to publish one, a lane unblocks
as soon as its own candidate lands rather than when the slowest peer
finishes, and an early row's merge overlaps a peer still nominating a late
one. Zero is "not arrived", which is why ``_ipc_alloc`` zeroes once and
nothing ever re-zeroes.

The tag costs 8 bits out of the block id, capping context at ``2**24``
blocks, and the double buffer costs twice the candidates -- both cheap next
to the barrier they replace, which at batch 64 cost more than the whole merge
does.

Generations are counted per launch block in ``cp_gen``, a rank-local int32
array that is never mapped to anyone: a candidate carries its own generation,
so the only shared state is the candidates. That makes the one hard rule
here: every rank must reach this module, and the kernel, the same number of
times with the same shapes. Ranks that disagree are comparing tags from
different calls, and because the merge waits rather than checks, the failure
is a hang and not a wrong answer. The same reasoning fixes the kernel's grid
to the candidate extent instead of the batch.
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
