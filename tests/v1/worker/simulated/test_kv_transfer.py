# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for simulated KV offloading transfers (no engine)."""

import pytest

from tests.v1.worker.simulated.utils import StubConfig, sim_req, sim_step
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.worker.simulated.kv_transfer import (
    SimulatedOffloadingWorker,
    TransferRate,
    transfer_rates_from_log,
)
from vllm.v1.worker.simulated.timing import (
    ReplayTimingProvider,
    StepCost,
    StepLogWriter,
    VirtualClock,
)


def _gpu_spec(*group_sizes: int) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(
        list(range(sum(group_sizes))), group_sizes, [0] * len(group_sizes)
    )


def test_kv_transfers_queue_per_direction_behind_device_work():
    clock = VirtualClock()
    worker = SimulatedOffloadingWorker(
        clock,
        load=TransferRate(latency_s=0.0, bytes_per_s=100.0),
        store=TransferRate(latency_s=0.25, bytes_per_s=50.0),
        bytes_per_block=[10, 30],
        device_free_at=lambda: 0.5,
    )
    # 2 blocks of group 0 and 1 of group 1: 50 bytes.
    worker.submit_load(1, None, _gpu_spec(2, 1))  # 0.5 -> 1.0
    worker.submit_load(2, None, _gpu_spec(1, 0))  # 1.0 -> 1.1
    worker.submit_store(3, _gpu_spec(5, 0), None)  # 0.5 -> 1.75
    assert worker.next_completion() == pytest.approx(1.0)

    clock.sleep_until(1.05)
    [done] = worker.get_finished()
    assert (done.job_id, done.transfer_size) == (1, 50)
    assert done.transfer_time == pytest.approx(0.5)

    worker.wait({3})
    assert clock.now() == pytest.approx(1.75)
    assert sorted(r.job_id for r in worker.get_finished()) == [2, 3]
    assert worker.next_completion() is None


def test_kv_transfer_rates_fitted_from_log(tmp_path):
    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    writer.write(
        sim_step(0, sim_req("a", 1, 8, 8)), StepCost(device_s=1e-3), 0.0, 1e-3, []
    )
    load = TransferRate(latency_s=20e-6, bytes_per_s=40e9)
    for count, num_bytes in [(1, 1 << 20), (3, 3 << 22), (2, 5 << 24)]:
        seconds = count * load.latency_s + num_bytes / load.bytes_per_s
        writer.write_transfers(
            {"load": {"count": count, "bytes": num_bytes, "seconds": seconds}}, t=0.0
        )
    # Durations that fall with size: no latency fits, only the mean bandwidth.
    for num_bytes, seconds in [(1 << 20, 2e-3), (4 << 20, 1e-3)]:
        writer.write_transfers(
            {"store": {"count": 1, "bytes": num_bytes, "seconds": seconds}}, t=0.0
        )
    writer.close()

    fitted_load, fitted_store = transfer_rates_from_log(str(path))
    assert fitted_load.latency_s == pytest.approx(load.latency_s)
    assert fitted_load.bytes_per_s == pytest.approx(load.bytes_per_s)
    assert fitted_store.latency_s == 0.0
    assert fitted_store.bytes_per_s == pytest.approx((5 << 20) / 3e-3)
    # The step providers skip the transfer records.
    assert len(ReplayTimingProvider(StubConfig(), path=str(path)).exact) == 1


def test_kv_transfer_rates_without_stores(tmp_path):
    from vllm.config.simulation import KVTransferTiming
    from vllm.v1.worker.simulated.kv_transfer import kv_transfer_rates

    timing = KVTransferTiming(load_gbps=2.0, load_latency_us=5.0)
    load, store = kv_transfer_rates(timing)
    assert (load.bytes_per_s, load.latency_s, store) == (2e9, pytest.approx(5e-6), None)
    # A disaggregated decode instance's recording has reads only.
    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    for num_bytes in (1 << 20, 4 << 20):
        writer.write_transfers(
            {"load": {"count": 1, "bytes": num_bytes, "seconds": num_bytes / 1e10}},
            t=0.0,
        )
    writer.close()
    load, store = transfer_rates_from_log(str(path))
    assert load.bytes_per_s == pytest.approx(1e10) and store is None
