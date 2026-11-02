# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StandaloneRuntime against a real umbp_standalone_server on ROCm GPUs."""

import importlib.util
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from tests.utils import multi_gpu_marks
from tests.v1.kv_connector.umbp_test_utils import mori_has_umbp
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    KVRange,
    RankTopology,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    StandaloneRuntime,
    UMBPRuntimeConfig,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not (current_platform.is_rocm() and mori_has_umbp()),
        reason="requires ROCm and MORI built with UMBP",
    ),
    *multi_gpu_marks(num_gpus=2),
]

_LAYER_BYTES = 1 << 20
_KEYS = ("standalone-block-0", "standalone-block-1")


def _server_binary() -> str:
    found = shutil.which("umbp_standalone_server")
    if found:
        return found
    mori_spec = importlib.util.find_spec("mori")
    assert mori_spec is not None and mori_spec.origin is not None
    candidate = Path(mori_spec.origin).parent / "umbp_standalone_server"
    if not candidate.exists():
        pytest.skip("umbp_standalone_server is not installed")
    return str(candidate)


@pytest.fixture
def server(tmp_path):
    address = f"unix://{tmp_path}/umbp.grpc.sock"
    env = dict(
        os.environ,
        UMBP_STANDALONE_ADDRESS=address,
        UMBP_DRAM_CAPACITY=str(256 * 1024 * 1024),
    )
    with open(tmp_path / "server.log", "w") as log:
        process = subprocess.Popen(
            [_server_binary()], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            yield SimpleServer(address, process, tmp_path / "server.log")
        finally:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()


class SimpleServer:
    def __init__(self, address: str, process: subprocess.Popen, log: Path):
        self.address = address
        self.process = process
        self.log = log

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(10)


def _runtime(address: str) -> StandaloneRuntime:
    return StandaloneRuntime(
        UMBPRuntimeConfig(
            "standalone",
            {"endpoint": address, "startup_timeout_ms": 20000, "num_workers": 2},
        )
    )


def _pattern(device: str) -> torch.Tensor:
    # Two layers, each holding two blocks; the object for block b is the two
    # layers' block b placed back to back.
    base = torch.arange(4 * _LAYER_BYTES, dtype=torch.int64) * 7919 % 251
    return base.to(torch.uint8).reshape(2, 2, _LAYER_BYTES).to(device)


def _plans(cache: torch.Tensor, block_offset: int = 0) -> list[BlockTransferPlan]:
    plans = []
    for block, key in enumerate(_KEYS):
        ranges = tuple(
            KVRange(
                f"layer{layer}",
                0,
                block + block_offset,
                cache[layer, block + block_offset].data_ptr(),
                _LAYER_BYTES,
                _LAYER_BYTES,
                layer * _LAYER_BYTES,
            )
            for layer in range(2)
        )
        plans.append(BlockTransferPlan(key, block + block_offset, ranges=ranges))
    return plans


def _producer(address: str, done) -> None:
    torch.accelerator.set_device_index(0)
    worker = _runtime(address).create_worker_handle("ns", RankTopology(), None)
    source = _pattern("cuda:0")
    worker.register_buffers({"kv": source})
    stored = worker.wait(worker.store(_plans(source)))
    assert stored.status is TransferJobStatus.COMPLETED, stored.error
    worker.publish(stored)
    worker.close()
    done.put(True)


def _store_from_another_process(address: str) -> None:
    context = mp.get_context("spawn")
    done = context.Queue()
    producer = context.Process(target=_producer, args=(address, done))
    producer.start()
    assert done.get(timeout=120)
    producer.join(30)
    assert producer.exitcode == 0


def test_objects_outlive_the_writer_and_restore_on_another_gpu(server):
    _store_from_another_process(server.address)

    torch.accelerator.set_device_index(1)
    runtime = _runtime(server.address)
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    assert list(scheduler.lookup([*_KEYS, "never-stored"])) == [True, True, False]

    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    destination = torch.zeros(2, 4, _LAYER_BYTES, dtype=torch.uint8, device="cuda:1")
    worker.register_buffers({"kv": destination})
    # Restore into blocks 2 and 3 so the destination offsets differ.
    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is TransferJobStatus.COMPLETED, loaded.error
    torch.accelerator.synchronize()

    expected = _pattern("cuda:1")
    assert torch.equal(destination[:, 2:4], expected)
    assert torch.count_nonzero(destination[:, 0:2]) == 0
    worker.close()
    scheduler.close()


def test_clear_removes_objects_for_every_client(server):
    _store_from_another_process(server.address)
    torch.accelerator.set_device_index(1)
    scheduler = _runtime(server.address).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    assert list(scheduler.lookup(_KEYS)) == [True, True]

    assert scheduler.clear() is True

    assert list(scheduler.lookup(_KEYS)) == [False, False]
    scheduler.close()


def test_server_loss_degrades_to_misses_and_failed_transfers(server):
    _store_from_another_process(server.address)
    torch.accelerator.set_device_index(1)
    runtime = _runtime(server.address)
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    destination = torch.zeros(2, 4, _LAYER_BYTES, dtype=torch.uint8, device="cuda:1")
    worker.register_buffers({"kv": destination})

    server.kill()
    time.sleep(1)

    assert list(scheduler.lookup(_KEYS)) == [False, False]
    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is not TransferJobStatus.COMPLETED
    stored = worker.wait(worker.store(_plans(destination)))
    assert stored.status is not TransferJobStatus.COMPLETED
    assert scheduler.clear() is False
    worker.close()
    scheduler.close()


_STOPPED_SERVER = """
import os
import signal
import sys
import time

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    KVRange,
    RankTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    StandaloneRuntime,
    UMBPRuntimeConfig,
)

address, server_pid, scenario = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.accelerator.set_device_index(0)
options = {"endpoint": address, "startup_timeout_ms": 20000, "timeout_ms": 200}
worker = StandaloneRuntime(
    UMBPRuntimeConfig("standalone", options)
).create_worker_handle("ns", RankTopology(), None)
source = torch.arange(1 << 20, dtype=torch.uint8, device="cuda:0")
worker.register_buffers({"kv": source})
os.kill(server_pid, signal.SIGSTOP)
if scenario == "store":
    size = source.numel()
    kv_range = KVRange("kv", 0, 0, source.data_ptr(), size, size, 0)
    job = worker.store([BlockTransferPlan("stuck", 0, ranges=(kv_range,))])
    time.sleep(0.5)
    assert worker.poll(job) is None
worker.close()
print("WORKER_CLOSE_RETURNED", flush=True)
"""


@pytest.mark.parametrize("scenario", ["store", "idle"])
def test_process_exits_while_the_server_stays_stopped(server, scenario):
    """A server that never resumes must not hold up close() or process exit."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _STOPPED_SERVER,
                server.address,
                str(server.process.pid),
                scenario,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        os.kill(server.process.pid, signal.SIGCONT)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "WORKER_CLOSE_RETURNED" in result.stdout
