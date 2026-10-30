# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-to-consumer TP mapping for MoRIIO WRITE transfers.

In WRITE mode every prefill (producer) rank pushes its own shard of the KV
cache and recurrent state into the decode (consumer) ranks that need it:

* consumer TP > producer TP (fan-out): producer rank ``p`` feeds consumer ranks
  ``p * r .. p * r + r - 1``. MLA latents and replicated KV heads are copied
  whole to each of them; split KV heads and mamba state are sliced, and each
  consumer gets its slice.
* producer TP > consumer TP (fan-in): consumer rank ``d`` is fed by producer
  ranks ``d * r .. d * r + r - 1``. Split KV heads and mamba state are
  gathered: each producer writes its shard at its offset inside the consumer's
  block. Copies that would be identical (MLA, replicated heads) are written by
  one producer only.

Heads follow vLLM's sharding: with ``tp_size <= num_kv_heads`` rank ``i`` holds
heads ``[i * n, (i + 1) * n)`` with ``n = num_kv_heads // tp_size``; with more
ranks than heads every rank holds the single head ``i * num_kv_heads //
tp_size``.

A consumer with decode context parallelism (DCP) shards the tokens of its MLA
caches instead. A producer with the same DCP size holds the same shard on the
same rank. A producer without DCP writes each consumer DCP rank
(``tp_rank % dcp_size``) the tokens it holds: with ``I =
cp_kv_cache_interleave_size``, token ``t`` lives on DCP rank ``(t // I) %
dcp_size`` at position ``p = (t // (I * dcp_size)) * I + t % I`` of that rank's
share, i.e. in its block ``p // block_size``. ``I`` divides the block size, so
each group of ``I`` tokens is contiguous on both sides and is one write
(``dcp_write_runs``); with ``I == block_size`` a group is a whole block.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import numpy as np

from vllm.utils.math_utils import cdiv


@dataclass(frozen=True)
class WriteTarget:
    """What one producer rank writes into one consumer rank."""

    tp_rank: int
    # Attention: copy ``num_heads`` heads starting at ``local_head`` of the
    # producer block into the consumer block starting at ``remote_head``.
    # ``num_heads == 0`` means another producer writes this identical copy.
    local_head: int
    remote_head: int
    num_heads: int
    # Mamba state: the producer's state is ``local_slices`` equal slices of
    # which ``local_slice`` is sent; it lands in slot ``remote_slice`` of the
    # consumer's ``remote_slices`` equal slices.
    local_slice: int
    local_slices: int
    remote_slice: int
    remote_slices: int
    # Producer ranks writing into this consumer rank, each sending one
    # ``write_done``; the consumer has its KV once all of them arrived.
    num_writers: int


@dataclass(frozen=True)
class MoRIIOWriteTPMapping:
    """Write targets of one producer rank for one consumer engine."""

    producer_tp_size: int
    consumer_tp_size: int
    local_num_heads: int
    remote_num_heads: int
    # Consumer ranks this producer writes to. Empty when another producer
    # writes everything this one holds (MLA or replicated heads, fan-in).
    targets: tuple[WriteTarget, ...]

    @property
    def target_tp_ranks(self) -> tuple[int, ...]:
        return tuple(target.tp_rank for target in self.targets)

    def copies_whole_block(self, target: WriteTarget) -> bool:
        """The consumer block is a byte copy of the producer block."""
        return target.num_heads == self.local_num_heads == self.remote_num_heads


def _rank_heads(tp_rank: int, tp_size: int, total_num_kv_heads: int) -> range:
    if tp_size <= total_num_kv_heads:
        per_rank = total_num_kv_heads // tp_size
        return range(tp_rank * per_rank, (tp_rank + 1) * per_rank)
    head = tp_rank * total_num_kv_heads // tp_size
    return range(head, head + 1)


def validate_write_tp_topology(
    producer_tp_size: int,
    consumer_tp_size: int,
    total_num_kv_heads: int,
    is_mla: bool,
) -> None:
    """Reject TP pairs that the WRITE mapping cannot express."""
    if producer_tp_size <= 0 or consumer_tp_size <= 0:
        raise ValueError("TP sizes must be positive")
    larger = max(producer_tp_size, consumer_tp_size)
    smaller = min(producer_tp_size, consumer_tp_size)
    if larger % smaller != 0:
        raise ValueError(
            "MoRIIO WRITE needs one TP size to be a multiple of the other, got "
            f"producer tp_size={producer_tp_size}, consumer "
            f"tp_size={consumer_tp_size}"
        )
    if is_mla or producer_tp_size == consumer_tp_size:
        return
    if total_num_kv_heads <= 0:
        raise ValueError("total_num_kv_heads must be positive")
    for tp_size in (producer_tp_size, consumer_tp_size):
        if (tp_size <= total_num_kv_heads and total_num_kv_heads % tp_size) or (
            tp_size > total_num_kv_heads and tp_size % total_num_kv_heads
        ):
            raise ValueError(
                f"MoRIIO WRITE cannot shard {total_num_kv_heads} KV heads over "
                f"tp_size={tp_size}"
            )


def _producer_targets(
    producer_tp_rank: int,
    producer_tp_size: int,
    consumer_tp_size: int,
    num_heads: int,
    has_mamba: bool,
) -> list[WriteTarget]:
    local_heads = _rank_heads(producer_tp_rank, producer_tp_size, num_heads)
    if consumer_tp_size >= producer_tp_size:
        ratio = consumer_tp_size // producer_tp_size
        targets = []
        for j in range(ratio):
            tp_rank = producer_tp_rank * ratio + j
            remote_heads = _rank_heads(tp_rank, consumer_tp_size, num_heads)
            targets.append(
                WriteTarget(
                    tp_rank=tp_rank,
                    local_head=remote_heads.start - local_heads.start,
                    remote_head=0,
                    num_heads=len(remote_heads),
                    local_slice=j,
                    local_slices=ratio,
                    remote_slice=0,
                    remote_slices=1,
                    num_writers=1,
                )
            )
        return targets

    ratio = producer_tp_size // consumer_tp_size
    tp_rank = producer_tp_rank // ratio
    remote_heads = _rank_heads(tp_rank, consumer_tp_size, num_heads)
    # The first producer holding a head writes it; the others hold an
    # identical copy and skip the attention write.
    group = range(tp_rank * ratio, (tp_rank + 1) * ratio)
    first_holder = next(
        p
        for p in group
        if _rank_heads(p, producer_tp_size, num_heads).start == local_heads.start
    )
    writes_attention = first_holder == producer_tp_rank
    if not writes_attention and not has_mamba:
        return []
    writers = sum(
        1
        for p in group
        if has_mamba
        or _rank_heads(p, producer_tp_size, num_heads).start
        not in {
            _rank_heads(q, producer_tp_size, num_heads).start
            for q in range(group.start, p)
        }
    )
    return [
        WriteTarget(
            tp_rank=tp_rank,
            local_head=0,
            remote_head=local_heads.start - remote_heads.start,
            num_heads=len(local_heads) if writes_attention else 0,
            local_slice=0,
            local_slices=1,
            remote_slice=producer_tp_rank % ratio,
            remote_slices=ratio,
            num_writers=writers,
        )
    ]


@cache
def compute_write_tp_mapping(
    producer_tp_rank: int,
    producer_tp_size: int,
    consumer_tp_size: int,
    total_num_kv_heads: int,
    is_mla: bool,
    has_mamba: bool,
) -> MoRIIOWriteTPMapping:
    """Map one producer rank onto the consumer ranks it writes."""
    validate_write_tp_topology(
        producer_tp_size, consumer_tp_size, total_num_kv_heads, is_mla
    )
    if not 0 <= producer_tp_rank < producer_tp_size:
        raise ValueError(
            f"producer_tp_rank {producer_tp_rank} must be in [0, {producer_tp_size})"
        )
    num_heads = 1 if is_mla else total_num_kv_heads
    first_remote_rank = producer_tp_rank * consumer_tp_size // producer_tp_size
    return MoRIIOWriteTPMapping(
        producer_tp_size=producer_tp_size,
        consumer_tp_size=consumer_tp_size,
        local_num_heads=len(_rank_heads(producer_tp_rank, producer_tp_size, num_heads)),
        remote_num_heads=len(
            _rank_heads(first_remote_rank, consumer_tp_size, num_heads)
        ),
        targets=tuple(
            _producer_targets(
                producer_tp_rank,
                producer_tp_size,
                consumer_tp_size,
                num_heads,
                has_mamba,
            )
        ),
    )


def validate_peer_dcp_layout(
    local_dcp_size: int,
    local_interleave_size: int,
    peer_dcp_size: int | None,
    peer_interleave_size: int | None,
    *,
    writes_into_peer: bool,
    block_size: int,
    is_mla: bool,
    whole_block_layer: str | None = None,
) -> None:
    """Refuse a peer whose DCP layout this worker cannot transfer KV with.

    Equal DCP sizes transfer rank to rank. A WRITE producer without DCP can
    also write into a DCP consumer, one write per group of interleaved tokens;
    ``whole_block_layer`` names a layer whose blocks can only be written whole,
    which needs the consumer's interleave to be the block size. A peer that
    does not advertise its layout predates these checks and is accepted as
    before.
    """
    if peer_dcp_size is None:
        return
    if peer_dcp_size == local_dcp_size:
        if (
            local_dcp_size > 1
            and peer_interleave_size is not None
            and peer_interleave_size != local_interleave_size
        ):
            raise ValueError(
                "MoRIIO needs the same --cp-kv-cache-interleave-size on both "
                f"instances with decode_context_parallel_size={local_dcp_size}: "
                f"{local_interleave_size} here, {peer_interleave_size} on the peer"
            )
        return
    if not (writes_into_peer and local_dcp_size == 1):
        raise ValueError(
            "MoRIIO cannot transfer KV between decode_context_parallel_size="
            f"{local_dcp_size} here and {peer_dcp_size} on the peer. Use the same "
            "size on both instances, or a prefill instance without decode "
            "context parallelism in WRITE mode"
        )
    if not is_mla:
        raise ValueError(
            "MoRIIO WRITE from a prefill instance without decode context "
            "parallelism into a decode instance with "
            f"decode_context_parallel_size={peer_dcp_size} supports MLA models only"
        )
    if peer_interleave_size is None or block_size % peer_interleave_size:
        raise ValueError(
            "MoRIIO WRITE into a decode instance with decode context parallelism "
            "needs a --cp-kv-cache-interleave-size that divides the block size "
            f"{block_size}; it uses {peer_interleave_size}"
        )
    if whole_block_layer is not None and peer_interleave_size < block_size:
        raise ValueError(
            f"MoRIIO writes layer {whole_block_layer} in whole blocks only, so "
            "the decode instance with decode_context_parallel_size="
            f"{peer_dcp_size} needs --cp-kv-cache-interleave-size {block_size} "
            f"(the block size); it uses {peer_interleave_size}"
        )


class DCPWriteRuns(NamedTuple):
    """First token of each run: producer block id and slot, consumer block id
    and slot."""

    local_blocks: np.ndarray
    local_slots: np.ndarray
    remote_blocks: np.ndarray
    remote_slots: np.ndarray


def dcp_write_runs(
    local_block_ids: Sequence[int],
    remote_block_ids: Sequence[int],
    block_size: int,
    dcp_size: int,
    interleave_size: int,
    dcp_rank: int,
) -> DCPWriteRuns:
    """Runs of ``interleave_size`` producer tokens that consumer DCP rank
    ``dcp_rank`` holds, in the rank's token order.

    A rank that holds none of them (a prompt shorter than ``dcp_size`` runs)
    gets empty runs and must still be sent write_done.
    """
    runs = np.arange(
        dcp_rank, len(local_block_ids) * block_size // interleave_size, dcp_size
    )
    tokens = runs * interleave_size
    positions = runs // dcp_size * interleave_size
    needed = cdiv(len(runs) * interleave_size, block_size)
    if len(remote_block_ids) < needed:
        raise ValueError(
            f"DCP rank {dcp_rank}/{dcp_size} of the consumer needs {needed} "
            f"blocks for {len(local_block_ids)} producer blocks, but it allocated "
            f"{len(remote_block_ids)}"
        )
    return DCPWriteRuns(
        np.asarray(local_block_ids, dtype=np.int64)[tokens // block_size],
        tokens % block_size,
        np.asarray(remote_block_ids, dtype=np.int64)[positions // block_size],
        positions % block_size,
    )
