# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public block-sidecar mappings for the native offloading connector."""

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorSidecarConfig,
    KVConnectorSidecarTransfer,
    KVConnectorSidecarTransfers,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def build_sidecar_config(
    spec: object,
) -> KVConnectorSidecarConfig | None:
    """Return the sidecar layout for supported offloading storage."""
    if not isinstance(spec, CPUOffloadingSpec):
        return None
    return KVConnectorSidecarConfig(
        num_connector_blocks=spec.num_blocks,
        blocks_per_connector_block=spec.blocks_per_chunk,
    )


def _divide_round_up(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def _normalize_transfer(
    gpu_spec: object,
    connector_spec: object,
    *,
    kv_group_id: int,
    expected_num_groups: int,
    blocks_per_connector_block: int,
) -> KVConnectorSidecarTransfer:
    if not isinstance(gpu_spec, GPULoadStoreSpec):
        raise RuntimeError(f"expected GPULoadStoreSpec, got {type(gpu_spec).__name__}")
    if not isinstance(connector_spec, CPULoadStoreSpec):
        raise RuntimeError(
            f"expected CPULoadStoreSpec, got {type(connector_spec).__name__}"
        )

    group_sizes = gpu_spec.group_sizes
    block_indices = gpu_spec.block_indices
    connector_blocks_per_group = [
        _divide_round_up(
            int(group_size) + int(block_indices[group_id]) % blocks_per_connector_block,
            blocks_per_connector_block,
        )
        for group_id, group_size in enumerate(group_sizes)
    ]
    if (
        len(group_sizes) != expected_num_groups
        or sum(group_sizes) != len(gpu_spec.block_ids)
        or sum(connector_blocks_per_group) != len(connector_spec.block_ids)
    ):
        raise RuntimeError(
            "KV block sidecar transfer violates the group-major flat-order "
            f"contract: group_sizes={list(group_sizes)}, "
            f"block_indices={list(block_indices)}, kv_group_id={kv_group_id}, "
            f"blocks_per_connector_block={blocks_per_connector_block}, "
            f"len(gpu)={len(gpu_spec.block_ids)}, "
            f"len(connector)={len(connector_spec.block_ids)}, "
            f"expected_connector={sum(connector_blocks_per_group)}, "
            f"expected_num_groups={expected_num_groups}"
        )

    gpu_block_offset = int(sum(group_sizes[:kv_group_id]))
    num_gpu_blocks = int(group_sizes[kv_group_id])
    gpu_block_ids = np.asarray(
        gpu_spec.block_ids[gpu_block_offset : gpu_block_offset + num_gpu_blocks]
    )
    if num_gpu_blocks == 0:
        empty = np.empty(0, dtype=np.int64)
        return KVConnectorSidecarTransfer(empty, empty, empty.copy())

    connector_block_offset = sum(connector_blocks_per_group[:kv_group_id])
    first_offset = int(block_indices[kv_group_id]) % blocks_per_connector_block
    block_positions = np.arange(num_gpu_blocks, dtype=np.int64)
    connector_block_offsets = (
        first_offset + block_positions
    ) % blocks_per_connector_block
    local_connector_indices = (
        first_offset + block_positions
    ) // blocks_per_connector_block
    connector_block_ids = np.asarray(connector_spec.block_ids)[
        connector_block_offset + local_connector_indices
    ]
    return KVConnectorSidecarTransfer(
        gpu_block_ids,
        connector_block_ids,
        connector_block_offsets,
    )


def normalize_sidecar_transfers(
    metadata: OffloadingConnectorMetadata,
    *,
    config: KVConnectorSidecarConfig,
    kv_group_id: int,
    expected_num_groups: int,
) -> KVConnectorSidecarTransfers:
    """Convert native offloading jobs into the public sidecar contract."""

    def normalize(
        gpu_spec: object, connector_spec: object
    ) -> KVConnectorSidecarTransfer:
        return _normalize_transfer(
            gpu_spec,
            connector_spec,
            kv_group_id=kv_group_id,
            expected_num_groups=expected_num_groups,
            blocks_per_connector_block=config.blocks_per_connector_block,
        )

    loads = [
        normalize(job.dst_spec, job.src_spec) for job in metadata.load_jobs.values()
    ]
    stores = [
        normalize(job.src_spec, job.dst_spec) for job in metadata.store_jobs.values()
    ]
    return KVConnectorSidecarTransfers(loads=loads, stores=stores)
