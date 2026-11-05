# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI-backed EmbeddedRuntime GPU integration tests."""

import pytest
import torch

from tests.v1.kv_connector.umbp_test_utils import mori_has_umbp
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
    UMBPRuntimeConfig,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_has_umbp()),
    reason="requires ROCm and MORI built with UMBP",
)


@pytest.mark.parametrize("bulk_load", [False, True])
def test_mori_embedded_dram_eviction_and_restore(socket_dir, bulk_load):
    """Overflowing DRAM evicts old objects, and the newest one still restores."""
    object_size = 1 << 20
    num_objects = 16
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig(
            "embedded",
            {
                "capacity_bytes": 8 * object_size,
                "dram_high_watermark": 0.75,
                "dram_low_watermark": 0.5,
                "lookup_dir": socket_dir,
                "timeout_ms": 60000,
            },
        )
    )
    worker = runtime.create_worker_handle(
        "mori-eviction",
        RankTopology(),
        KVLayoutDescriptor(
            regions=(KVRegion("layer0", 0, object_size, object_size, 0),),
            topology=RankTopology(),
        ),
    )
    try:
        source = torch.empty((1, object_size), dtype=torch.uint8, device="cuda")
        destination = torch.zeros_like(source)
        worker.register_buffers({"layer0": source, "destination": destination})

        keys = [f"eviction-{index}" for index in range(num_objects)]
        for index, key in enumerate(keys):
            source.fill_(index)
            plan = BlockTransferPlan(
                key,
                index,
                ranges=(
                    KVRange(
                        "layer0",
                        0,
                        index,
                        source.data_ptr(),
                        object_size,
                        object_size,
                        0,
                    ),
                ),
            )
            stored = worker.wait(worker.store([plan]))
            if stored.status is TransferJobStatus.COMPLETED:
                worker.publish(stored)

        hits = list(worker.batch_exists(keys))
        assert not all(hits), "the stores never exceeded DRAM capacity"
        latest = max(index for index, hit in enumerate(hits) if hit)

        if bulk_load:
            worker.register_buffers({"layer0": destination})
            job = worker.load_blocks([BlockTransferPlan(keys[latest], 0, group_id=0)])
            assert job is not None
        else:
            job = worker.load(
                [
                    BlockTransferPlan(
                        keys[latest],
                        0,
                        ranges=(
                            KVRange(
                                "layer0",
                                0,
                                0,
                                destination.data_ptr(),
                                object_size,
                                object_size,
                                0,
                            ),
                        ),
                    )
                ]
            )
        loaded = worker.wait(job)
        assert loaded.status is TransferJobStatus.COMPLETED
        assert loaded.completed_bytes == object_size
        torch.accelerator.synchronize()
        assert torch.all(destination == latest)
    finally:
        worker.close()
