# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.sidecar import (
    build_sidecar_config,
    normalize_sidecar_transfers,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def _cpu_spec(
    spec_type: type[CPUOffloadingSpec] = CPUOffloadingSpec,
) -> CPUOffloadingSpec:
    spec = object.__new__(spec_type)
    spec.num_blocks = 17
    spec.blocks_per_chunk = 2
    return spec


def test_build_sidecar_config_supports_only_native_cpu_offload():
    assert build_sidecar_config(_cpu_spec()) is not None
    assert build_sidecar_config(_cpu_spec(TieringOffloadingSpec)) is None


def test_normalize_sidecar_transfers_preserves_direction():
    config = build_sidecar_config(_cpu_spec())
    assert config is not None

    load_gpu_spec = GPULoadStoreSpec(
        block_ids=[10, 11, 12, 20],
        group_sizes=[3, 1],
        block_indices=[1, 0],
    )
    load_connector_spec = CPULoadStoreSpec([100, 101, 200])
    store_gpu_spec = GPULoadStoreSpec(
        block_ids=[30, 31, 32, 40],
        group_sizes=[3, 1],
        block_indices=[0, 0],
    )
    store_connector_spec = CPULoadStoreSpec([300, 301, 400])
    metadata = OffloadingConnectorMetadata(
        load_jobs={
            1: TransferJob(
                req_id="load",
                src_spec=load_connector_spec,
                dst_spec=load_gpu_spec,
            ),
            3: TransferJob(
                req_id="load-2",
                src_spec=CPULoadStoreSpec([102, 201]),
                dst_spec=GPULoadStoreSpec(
                    block_ids=[13, 21],
                    group_sizes=[1, 1],
                    block_indices=[4, 2],
                ),
            ),
        },
        store_jobs={
            2: TransferJob(
                req_id="store",
                src_spec=store_gpu_spec,
                dst_spec=store_connector_spec,
            )
        },
    )

    transfers = normalize_sidecar_transfers(
        metadata,
        config=config,
        kv_group_id=0,
        expected_num_groups=2,
    )

    assert transfers.load is not None
    np.testing.assert_array_equal(
        transfers.load.gpu_block_ids,
        [10, 11, 12, 13],
    )
    np.testing.assert_array_equal(
        transfers.load.connector_block_ids,
        [100, 101, 101, 102],
    )
    np.testing.assert_array_equal(
        transfers.load.connector_block_offsets,
        [1, 0, 1, 0],
    )

    assert transfers.store is not None
    np.testing.assert_array_equal(
        transfers.store.gpu_block_ids,
        [30, 31, 32],
    )
    np.testing.assert_array_equal(
        transfers.store.connector_block_ids,
        [300, 300, 301],
    )
    np.testing.assert_array_equal(
        transfers.store.connector_block_offsets,
        [0, 1, 0],
    )


def test_normalize_sidecar_transfers_rejects_malformed_group_metadata():
    config = build_sidecar_config(_cpu_spec())
    assert config is not None

    gpu_spec = GPULoadStoreSpec.__new__(GPULoadStoreSpec)
    gpu_spec.block_ids = np.array([10, 20], dtype=np.int64)
    gpu_spec.group_sizes = [1, 1]
    gpu_spec.block_indices = [0]
    connector_spec = CPULoadStoreSpec([100, 200])
    metadata = OffloadingConnectorMetadata(
        load_jobs={
            1: TransferJob(
                req_id="load",
                src_spec=connector_spec,
                dst_spec=gpu_spec,
            )
        },
        store_jobs={},
    )

    with pytest.raises(RuntimeError, match="group-major flat-order contract"):
        normalize_sidecar_transfers(
            metadata,
            config=config,
            kv_group_id=0,
            expected_num_groups=2,
        )
