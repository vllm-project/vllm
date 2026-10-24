# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-side descriptor planning for the Mooncake pull connector.

`MooncakeConnectorWorker` turns a decoder's block table into source and
destination descriptors and writes them. A pull connector needs the same plan
with the two sides swapped, because the decoder reads the prefiller's bytes into
its own cache: the prefiller's blocks are the remote descriptors and the
decoder's blocks are the local ones. This module builds that plan from the same
registered regions.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    ReqId,
    TransferRegion,
    _block_ids_for_region,
    _can_coalesce_block_transfers,
    group_concurrent_contiguous,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

logger = init_logger(__name__)

TransferPlan = Callable[[TransferRegion, TransferRegion], tuple[bool, int, int, int]]


@dataclass(frozen=True)
class ReadRequest:
    """One request's blocks on both sides of a read.

    Attributes:
        key: Request id the caller uses to report failures.
        producer_block_ids: Per KV group, the blocks holding the KV to read.
        consumer_block_ids: Per KV group, the local blocks to read it into.

    """

    key: ReqId
    producer_block_ids: list[list[int]]
    consumer_block_ids: list[list[int]]


def _kernel_block_ids(
    block_ids: list[list[int]],
    group_specs: Sequence[KVCacheGroupSpec],
    blocks_per_logical: int,
) -> list[list[int]]:
    """Expand logical block ids to the kernel blocks serving them.

    A logical block served by several attention kernel blocks expands to all of
    them, while a Mamba/GDN state block keeps its logical id.
    """
    if blocks_per_logical == 1:
        return block_ids

    block_arange = np.arange(blocks_per_logical).reshape(1, -1)
    return [
        (
            (np.array(group)[:, None] * blocks_per_logical + block_arange)
            .reshape(-1)
            .tolist()
            if not isinstance(group_specs[index].kv_cache_spec, MambaSpec)
            else group
        )
        for index, group in enumerate(block_ids)
    ]


def _drop_mamba_placeholders(
    block_ids: list[list[int]], group_specs: Sequence[KVCacheGroupSpec]
) -> list[list[int]]:
    """Drop null blocks, which are prefix-cache placeholders without state."""
    return [
        (
            [block_id for block_id in group if block_id != NULL_BLOCK_ID]
            if isinstance(group_specs[index].kv_cache_spec, MambaSpec)
            else group
        )
        for index, group in enumerate(block_ids)
    ]


def plan_read_block_descriptors(
    requests: Sequence[ReadRequest],
    *,
    producer_regions: Sequence[TransferRegion],
    consumer_regions: Sequence[TransferRegion],
    group_specs: Sequence[KVCacheGroupSpec],
    producer_blocks_per_logical: int,
    consumer_blocks_per_logical: int,
    plan: TransferPlan,
) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
    """Plan one read per request into local and remote descriptor lists.

    Args:
        requests: Requests to read, each carrying both sides' block ids.
        producer_regions: Aligned regions of the prefiller's KV cache.
        consumer_regions: Aligned regions of this decoder's KV cache.
        group_specs: KV cache groups of this instance.
        producer_blocks_per_logical: Kernel blocks per logical block on the peer.
        consumer_blocks_per_logical: Kernel blocks per logical block locally.
        plan: Callback returning the transfer flag, the producer byte offset,
            the consumer byte offset and the copy length for one region pair.

    Returns:
        Local (decoder) descriptors, remote (prefiller) descriptors, lengths,
        the request ids that cannot be read, and the reason for those failures.

    """
    local_ptrs: list[int] = []
    remote_ptrs: list[int] = []
    lengths: list[int] = []
    err_reqs: list[ReqId] = []
    err_msg: str | None = None

    for request in requests:
        if not any(request.producer_block_ids) or not any(request.consumer_block_ids):
            # An empty read only releases the prefiller's blocks.
            continue

        if len(request.producer_block_ids) != len(request.consumer_block_ids):
            logger.error(
                "req %s: KV group count mismatch: producer=%d, consumer=%d",
                request.key,
                len(request.producer_block_ids),
                len(request.consumer_block_ids),
            )
            err_reqs.append(request.key)
            if err_msg is None:
                err_msg = "KV group count mismatch"
            continue

        assert len(request.consumer_block_ids) == len(group_specs), (
            "A read must carry one block list per local KV cache group."
        )

        producer_groups = _drop_mamba_placeholders(
            request.producer_block_ids, group_specs
        )
        consumer_groups = _drop_mamba_placeholders(
            request.consumer_block_ids, group_specs
        )

        trimmed_producer: list[list[int]] = []
        trimmed_consumer: list[list[int]] = []
        for producer_group, consumer_group in zip(producer_groups, consumer_groups):
            if len(producer_group) < len(consumer_group):
                logger.error(
                    "req %s: producer blocks(%d) < consumer blocks(%d) in a KV "
                    "cache group",
                    request.key,
                    len(producer_group),
                    len(consumer_group),
                )
                err_reqs.append(request.key)
                if err_msg is None:
                    err_msg = "producer num blocks less than consumer"
                break
            if len(producer_group) > len(consumer_group):
                # The decoder already holds a prefix, so its read covers the
                # last blocks of the prefiller's request.
                producer_group = (
                    producer_group[-len(consumer_group) :] if consumer_group else []
                )
            trimmed_producer.append(producer_group)
            trimmed_consumer.append(consumer_group)
        else:
            producer_groups, consumer_groups = trimmed_producer, trimmed_consumer

        if err_reqs and err_reqs[-1] == request.key:
            continue

        producer_groups = _kernel_block_ids(
            producer_groups, group_specs, producer_blocks_per_logical
        )
        consumer_groups = _kernel_block_ids(
            consumer_groups, group_specs, consumer_blocks_per_logical
        )

        for producer_region, consumer_region in zip(producer_regions, consumer_regions):
            assert producer_region.group_index == consumer_region.group_index, (
                "Aligned Mooncake transfer regions must belong to the same KV group."
            )
            producer_block_ids = _block_ids_for_region(
                producer_groups,
                producer_region.group_index,
                producer_region.shared_group_ids,
            )
            consumer_block_ids = _block_ids_for_region(
                consumer_groups,
                consumer_region.group_index,
                consumer_region.shared_group_ids,
            )
            if not consumer_block_ids:
                continue

            grouped_producer_blocks, grouped_consumer_blocks = (
                group_concurrent_contiguous(producer_block_ids, consumer_block_ids)
            )
            (
                should_transfer,
                producer_offset,
                consumer_offset,
                transfer_len,
            ) = plan(producer_region, consumer_region)
            if not should_transfer:
                # A replicated prefiller cache: another rank holds these bytes.
                continue

            assert producer_offset + transfer_len <= producer_region.kv_block_len, (
                "Computed read offset exceeds the producer KV block size."
            )
            assert consumer_offset + transfer_len <= consumer_region.kv_block_len, (
                "Computed read offset exceeds the consumer KV block size."
            )
            # Collapse one contiguous block group into a single larger
            # descriptor when the per-block copy is identical.
            can_coalesce = _can_coalesce_block_transfers(
                local_region_block_len=producer_region.block_len,
                remote_region_block_len=consumer_region.block_len,
                src_region_offset=producer_offset,
                dst_region_offset=consumer_offset,
                transfer_len=transfer_len,
            )

            for producer_blocks, consumer_blocks in zip(
                grouped_producer_blocks, grouped_consumer_blocks
            ):
                if can_coalesce:
                    remote_ptrs.append(
                        producer_region.base_addr
                        + producer_blocks[0] * producer_region.block_len
                        + producer_offset
                    )
                    local_ptrs.append(
                        consumer_region.base_addr
                        + consumer_blocks[0] * consumer_region.block_len
                        + consumer_offset
                    )
                    lengths.append(transfer_len * len(consumer_blocks))
                else:
                    for producer_block, consumer_block in zip(
                        producer_blocks, consumer_blocks
                    ):
                        remote_ptrs.append(
                            producer_region.base_addr
                            + producer_block * producer_region.block_len
                            + producer_offset
                        )
                        local_ptrs.append(
                            consumer_region.base_addr
                            + consumer_block * consumer_region.block_len
                            + consumer_offset
                        )
                        lengths.append(transfer_len)

    return local_ptrs, remote_ptrs, lengths, err_reqs, err_msg
