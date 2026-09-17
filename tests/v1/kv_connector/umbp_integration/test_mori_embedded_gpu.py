# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI-backed EmbeddedRuntime GPU integration tests."""

import multiprocessing as mp
import time
from typing import Any

import pytest
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
    UMBPRuntimeConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.embedded import (
    _MoriSchedulerHandle,
)

pytest.importorskip("mori.cpp")

_SIZE = 1 << 20
_NAMESPACE = "mori-embedded-tp2-pp2"


def _topology(rank: int) -> RankTopology:
    return RankTopology(
        tp_rank=rank % 2,
        tp_size=2,
        pp_rank=rank // 2,
        pp_size=2,
    )


def _gpu_worker(
    rank: int,
    lookup_dir: str,
    ready: mp.Queue,
    release: Any,
) -> None:
    torch.accelerator.set_device_index(rank)
    topology = _topology(rank)
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, _SIZE, _SIZE, 0),),
        topology=topology,
    )
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig(
            "embedded",
            {
                "capacity_bytes": 128 * 1024 * 1024,
                "lookup_dir": lookup_dir,
                "num_workers": 2,
            },
        )
    )
    worker = runtime.create_worker_handle(_NAMESPACE, topology, layout)
    source = torch.arange(_SIZE, dtype=torch.uint8, device=f"cuda:{rank}") + rank
    destination = torch.zeros_like(source)
    worker.register_buffers({"layer0": source})
    key = BlockIdentityCodec(
        UMBPNamespace(_NAMESPACE),
        tp_rank=topology.tp_rank,
        pp_rank=topology.pp_rank,
    ).key(b"shared-logical-block")
    store_plan = BlockTransferPlan(
        key,
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), _SIZE, _SIZE, 0),),
    )
    stored = worker.wait(worker.store([store_plan]))
    assert stored.status is TransferJobStatus.COMPLETED
    worker.publish(stored)

    worker.register_buffers({"layer0": destination})
    load_plan = BlockTransferPlan(
        key,
        1,
        ranges=(KVRange("layer0", 0, 1, destination.data_ptr(), _SIZE, _SIZE, 0),),
    )
    loaded = worker.wait(worker.load([load_plan]))
    assert loaded.status is TransferJobStatus.COMPLETED
    torch.accelerator.synchronize()
    assert torch.equal(source, destination)
    ready.put((rank, key))
    release.wait(60)
    worker.close()


@pytest.mark.skipif(
    not torch.accelerator.is_available() or torch.accelerator.device_count() < 4,
    reason="requires four ROCm GPUs",
)
def test_mori_embedded_tp2_pp2_gpu_roundtrip(tmp_path):
    context = mp.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    processes = [
        context.Process(
            target=_gpu_worker,
            args=(rank, str(tmp_path), ready, release),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    entries = sorted(ready.get(timeout=90) for _ in processes)
    scheduler = _MoriSchedulerHandle(
        _NAMESPACE,
        _topology(0),
        str(tmp_path),
    )
    deadline = time.monotonic() + 10
    hits = [False] * 4
    while time.monotonic() < deadline:
        hits = list(scheduler.lookup([key for _, key in entries]))
        if all(hits):
            break
        time.sleep(0.1)
    release.set()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    scheduler.close()
    assert all(hits)


@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="requires a ROCm GPU",
)
def test_mori_store_overlaps_gpu_compute(tmp_path):
    torch.accelerator.set_device_index(0)
    size = 64 << 20
    topology = RankTopology()
    layout = KVLayoutDescriptor(
        regions=(KVRegion("layer0", 0, size, size, 0),),
        topology=topology,
    )
    runtime = EmbeddedRuntime.from_config(
        UMBPRuntimeConfig(
            "embedded",
            {
                "capacity_bytes": 256 << 20,
                "lookup_dir": str(tmp_path),
                "num_workers": 2,
                "timeout_ms": 60000,
            },
        )
    )
    worker = runtime.create_worker_handle(
        "mori-compute-store-overlap", topology, layout
    )
    source = torch.arange(size, dtype=torch.uint8, device="cuda:0")
    worker.register_buffers({"layer0": source})
    plan = BlockTransferPlan(
        "overlap-key",
        0,
        ranges=(KVRange("layer0", 0, 0, source.data_ptr(), size, size, 0),),
    )
    left = torch.randn((4096, 4096), dtype=torch.float16, device="cuda:0")
    right = torch.randn_like(left)
    compute_stream = torch.Stream(device="cuda:0")
    torch.mm(left, right)
    torch.accelerator.synchronize()

    store_start = time.perf_counter()
    store_job = worker.store([plan])
    compute_start = time.perf_counter()
    with compute_stream:
        compute_begin = torch.Event(enable_timing=True)
        compute_end = torch.Event(enable_timing=True)
        compute_begin.record()
        for _ in range(32):
            output = torch.mm(left, right)
        compute_end.record()
    stored = worker.wait(store_job)
    store_end = time.perf_counter()
    compute_stream.synchronize()
    compute_end_time = time.perf_counter()

    overlap_seconds = min(store_end, compute_end_time) - max(store_start, compute_start)
    print(
        "store_s=",
        store_end - store_start,
        "compute_ms=",
        compute_begin.elapsed_time(compute_end),
        "overlap_s=",
        overlap_seconds,
    )
    assert stored.status is TransferJobStatus.COMPLETED
    assert output.is_cuda
    assert compute_begin.elapsed_time(compute_end) > 0
    assert overlap_seconds > 0
    worker.publish(stored)
    worker.close()
