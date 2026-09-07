# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    DirectionalTransferStats,
    OffloadingWorkerMetadata,
    TransferStats,
)

pytestmark = pytest.mark.cpu_test


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 2, 7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={43: 1, 8: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 1, 7: 1, 43: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1, 8: 1})
    meta3 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 8: 1})
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_jobs == {42: 3, 43: 2, 7: 2, 8: 2}


def test_aggregate_transfer_stats():
    meta1 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=10, time=0.5, sizes=[10])
        )
    )
    meta2 = OffloadingWorkerMetadata(
        transfer_stats=TransferStats(
            load=DirectionalTransferStats(bytes=20, time=1.0, sizes=[20, 30])
        )
    )

    result = meta1.aggregate(meta2)

    assert result.transfer_stats.load.bytes == 30
    assert result.transfer_stats.load.time == 1.5
    assert result.transfer_stats.load.sizes == [10, 20, 30]
