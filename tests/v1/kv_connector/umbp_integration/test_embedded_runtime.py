# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedded runtime contract tests.

These tests use the explicit CPU memory backend.  The MORI-backed runtime
uses the same handles and plans, but requires the MORI UMBP extension and an
AMD GPU.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutDescriptor,
    KVRange,
    KVRegion,
    RankTopology,
    TransferJobStatus,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    EmbeddedRuntime,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
    UMBPRuntimeConfig,
)


def _runtime():
    return EmbeddedRuntime.from_config(
        UMBPRuntimeConfig("embedded", {"backend": "memory"})
    )


def _layout() -> KVLayoutDescriptor:
    return KVLayoutDescriptor(
        regions=(
            KVRegion(
                layer_name="layer0",
                group_id=0,
                block_stride=16,
                block_bytes=16,
                object_offset=0,
            ),
        ),
        topology=RankTopology(),
    )


def _plan(key: str, tensor: torch.Tensor, block_id: int = 0):
    return BlockTransferPlan(
        key=key,
        block_id=block_id,
        ranges=(
            KVRange(
                layer_name="layer0",
                group_id=0,
                block_id=block_id,
                base_address=tensor.data_ptr(),
                stride=tensor.numel(),
                length=tensor.numel(),
                object_offset=0,
            ),
        ),
    )


def test_embedded_store_is_invisible_until_publish():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    plan = _plan("atomic-key", source)

    worker.register_buffers({"layer0": source})
    job = worker.store([plan])

    assert scheduler.lookup(["atomic-key"]) == [False]
    assert worker.wait(job).status is TransferJobStatus.COMPLETED

    worker.publish(job)

    assert scheduler.lookup(["atomic-key"]) == [True]
    worker.close()
    scheduler.close()


def test_embedded_store_load_roundtrip_restores_bytes():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)
    plan = _plan("roundtrip-key", source, block_id=3)

    worker.register_buffers({"layer0": source})
    store_job = worker.store([plan])
    worker.publish(worker.wait(store_job))

    load_plan = _plan("roundtrip-key", destination, block_id=7)
    load_job = worker.load([load_plan])
    result = worker.wait(load_job)

    assert result.status is TransferJobStatus.COMPLETED
    assert torch.equal(source, destination)
    worker.close()
    scheduler.close()


def test_embedded_missing_object_is_a_load_failure():
    runtime = _runtime()
    layout = _layout()
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    destination = torch.zeros(16, dtype=torch.uint8)

    worker.register_buffers({"layer0": destination})
    result = worker.wait(worker.load([_plan("missing-key", destination, 9)]))

    assert result.status is TransferJobStatus.FAILED
    assert result.failed_block_ids == {9}
    worker.close()


def test_embedded_batch_isolates_failed_objects():
    runtime = _runtime()
    layout = _layout()
    scheduler = runtime.create_scheduler_handle("test", RankTopology(), layout)
    worker = runtime.create_worker_handle("test", RankTopology(), layout)
    source = torch.arange(16, dtype=torch.uint8)
    destination = torch.zeros_like(source)

    worker.register_buffers({"layer0": source})
    stored = _plan("present-key", source, block_id=1)
    store_job = worker.store([stored])
    worker.publish(worker.wait(store_job))

    present = _plan("present-key", destination, block_id=4)
    missing = _plan("missing-key", destination, block_id=5)
    result = worker.wait(worker.load([present, missing]))

    assert result.status is TransferJobStatus.FAILED
    assert result.completed_keys == {"present-key"}
    assert result.failed_block_ids == {5}
    assert scheduler.lookup(["present-key", "missing-key"]) == [True, False]
    assert torch.equal(source, destination)
    worker.close()
    scheduler.close()


def test_rank_local_keys_are_distinct_for_tp_pp_dcp():
    topology = RankTopology(
        tp_size=2,
        pp_size=2,
        dcp_size=2,
    )
    codec = BlockIdentityCodec(UMBPNamespace("rank-test"))
    keys = codec.keys_for_topology(b"hash", topology, [0])

    assert len(keys) == 8
    assert len(set(keys)) == 8
    assert all(":tp" in key and ":dcp" in key and ":pp" in key for key in keys)
