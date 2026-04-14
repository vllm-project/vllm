# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)

pytestmark = pytest.mark.cpu_test


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1},
        completed_load_jobs={7: 1},
    )
    meta2 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1},
        completed_load_jobs={7: 1},
    )
    result = meta1.aggregate(meta2)
    assert result.completed_store_jobs == {42: 2}
    assert result.completed_load_jobs == {7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1},
        completed_load_jobs={7: 1},
    )
    meta2 = OffloadingWorkerMetadata(
        completed_store_jobs={43: 1},
        completed_load_jobs={8: 1},
    )
    result = meta1.aggregate(meta2)
    assert result.completed_store_jobs == {42: 1, 43: 1}
    assert result.completed_load_jobs == {7: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1, 43: 1},
        completed_load_jobs={7: 1},
    )
    meta2 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1},
        completed_load_jobs={7: 1, 8: 1},
    )
    meta3 = OffloadingWorkerMetadata(
        completed_store_jobs={42: 1, 43: 1},
        completed_load_jobs={8: 1},
    )
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_store_jobs == {42: 3, 43: 2}
    assert result.completed_load_jobs == {7: 2, 8: 2}
