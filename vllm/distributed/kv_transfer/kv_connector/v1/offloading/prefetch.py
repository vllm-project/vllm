# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch hints: bring offloaded KV back into the GPU pool before its request.

A prefetch hint names block hashes whose KV already lives in an offloading tier.
The blocks are pinned in the GPU pool, filled by that tier, published to the
prefix cache, and handed back to the pool. A request that arrives later then
hits them on the ordinary prefix-cache path and never waits for the transfer.

Only the arithmetic lives here so it can be tested without an engine: which
hashes to pin, and which offload key owns each chunk.
"""

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PinnedBlockSet
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key

logger = init_logger(__name__)

PREFETCH_ACTION_TYPE = "agenthint.prefetch"
"""Action type of a prefetch hint. Payload is {"block_hashes": [<hex>, ...]}."""

PREFETCH_ACTION_VERSION = "1.0"

PREFETCH_REQ_ID = "kv-prefetch"
"""Request id of the manager context a prefetch transfer runs under.

Nothing request-scoped is read from it: the offloading manager only uses a
context to attach state and to map a key to its position in the prefix. A
distinct id keeps a prefetch out of any per-request bookkeeping.
"""


@dataclass(slots=True, frozen=True)
class PrefetchPlan:
    """What a hint asks for, in the terms a transfer needs."""

    block_hashes: tuple[bytes, ...]
    """Per-block hashes to pin, in token order. Whole chunks only."""

    keys_by_group: dict[int, tuple[OffloadKey, ...]]
    """Offload keys to load, one per chunk, per KV cache group."""

    num_chunks: int

    def dst_block_ids(self, pinned: list[PinnedBlockSet]) -> list[int]:
        """GPU block ids of a reservation, group-major like the load spec."""
        return [block.block_id for pinned_set in pinned for block in pinned_set.blocks]

    def group_sizes(self, pinned: list[PinnedBlockSet]) -> list[int]:
        return [len(pinned_set.blocks) for pinned_set in pinned]

    def block_indices(self) -> list[int]:
        """First logical block index of each group.

        A prefetch always starts at a chunk boundary, so no partial head has to
        be skipped.
        """
        return [0] * len(self.keys_by_group)


@dataclass(slots=True)
class PrefetchJob:
    """Scheduler-side state of one prefetch transfer."""

    plan: PrefetchPlan
    pinned: list[PinnedBlockSet]
    req_context: ReqContext
    """The context its load was prepared under; `complete_load` needs it back."""


def decode_prefetch_hashes(
    envelope: KvHintsEnvelope | None,
) -> tuple[bytes, ...]:
    """Block hashes named by a prefetch action of this envelope.

    Returns an empty tuple when the envelope carries no prefetch action, which
    is the common case: hints are usually for the request itself.
    """
    if not isinstance(envelope, KvHintsEnvelope):
        return ()
    for action in envelope.actions:
        if action.action_type != PREFETCH_ACTION_TYPE:
            continue
        if action.action_version != PREFETCH_ACTION_VERSION:
            logger.warning_once(
                "Ignoring prefetch hint with unsupported action_version %s",
                action.action_version,
            )
            continue
        raw = action.payload.get("block_hashes")
        if not isinstance(raw, list) or not raw:
            logger.warning_once("Ignoring prefetch hint without block_hashes")
            continue
        try:
            return tuple(bytes.fromhex(str(entry)) for entry in raw)
        except ValueError:
            logger.warning_once("Ignoring prefetch hint with a malformed hash")
    return ()


def plan_prefetch(
    block_hashes: tuple[bytes, ...],
    *,
    hashes_per_chunk: int,
    blocks_per_chunk: int,
    group_ids: tuple[int, ...],
) -> PrefetchPlan | None:
    """Turn per-block hashes into what a transfer needs, or None.

    A chunk is keyed by the hash of its last block, so a prefix of `n` chunks
    names hashes `stride-1`, `2*stride-1`, ... The trailing partial chunk, if
    any, is dropped: its KV was never keyed and can never be a hit.
    """
    if not block_hashes or not group_ids or hashes_per_chunk <= 0:
        return None
    if hashes_per_chunk != blocks_per_chunk:
        # Pinning is per hash and loading is per block, so the two granularities
        # have to line up. Mismatched sizes mean one hash covers several blocks,
        # which this reservation cannot express.
        logger.warning_once(
            "Prefetch declined: hashes_per_chunk %d != blocks_per_chunk %d",
            hashes_per_chunk,
            blocks_per_chunk,
        )
        return None
    stride = hashes_per_chunk
    num_chunks = len(block_hashes) // stride
    if num_chunks <= 0:
        return None
    kept = block_hashes[: num_chunks * stride]
    keys_by_group = {
        group_id: tuple(
            make_offload_key(kept[stride * (chunk + 1) - 1], group_id)
            for chunk in range(num_chunks)
        )
        for group_id in group_ids
    }
    return PrefetchPlan(
        block_hashes=kept, keys_by_group=keys_by_group, num_chunks=num_chunks
    )


def truncate_plan(plan: PrefetchPlan, num_chunks: int) -> PrefetchPlan | None:
    """Keep the leading `num_chunks` chunks of a plan, or None if there are none.

    Used once lookups have shown how far the tier can serve: a load has to be a
    contiguous prefix, so only that far is worth reserving.
    """
    if num_chunks <= 0:
        return None
    if num_chunks >= plan.num_chunks:
        return plan
    stride = len(plan.block_hashes) // plan.num_chunks
    return PrefetchPlan(
        block_hashes=plan.block_hashes[: num_chunks * stride],
        keys_by_group={
            group_id: keys[:num_chunks]
            for group_id, keys in plan.keys_by_group.items()
        },
        num_chunks=num_chunks,
    )
