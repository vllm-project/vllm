# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DistributedRuntime against a real umbp_master and RDMA on ROCm GPUs.

The producer and consumer are separate pool nodes on one host, so every
transfer between them takes the remote (RDMA) path a second host would.
"""

import mmap
import multiprocessing as mp
import os
import random
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockTransferPlan,
    KVRange,
    RankTopology,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import (
    DistributedRuntime,
    UMBPRuntimeConfig,
)

mori = pytest.importorskip("mori")
pytest.importorskip("mori.cpp")


def _has_active_rdma_port() -> bool:
    for state in Path("/sys/class/infiniband").glob("*/ports/*/state"):
        if "ACTIVE" in state.read_text():
            return True
    return False


pytestmark = [
    pytest.mark.skipif(
        not torch.accelerator.is_available() or torch.accelerator.device_count() < 2,
        reason="requires two ROCm GPUs",
    ),
    pytest.mark.skipif(not _has_active_rdma_port(), reason="requires an RDMA NIC"),
]

_LAYER_BYTES = 1 << 20
_KEYS = ("distributed-block-0", "distributed-block-1")
_HEARTBEAT_TTL_S = 2


def _free_port(span: int = 1) -> int:
    """Return a port p such that p .. p + span - 1 are all free.

    Ports come from below Linux's ephemeral range, which outgoing connections
    draw their local ports from, so a port found free stays free until bound.
    """
    rng = random.Random()
    for _ in range(100):
        base = rng.randrange(20000, 32000 - span)
        sockets = []
        try:
            for port in range(base, base + span):
                probe = socket.socket()
                sockets.append(probe)
                probe.bind(("0.0.0.0", port))
            return base
        except OSError:
            continue
        finally:
            for probe in sockets:
                probe.close()
    raise RuntimeError(f"no run of {span} free ports")


# Workers bind peer_service_port plus their physical GPU index.
_PEER_PORT_SPAN = 8


class Master:
    def __init__(self, tmp_path: Path) -> None:
        self.port = _free_port()
        self.address = f"127.0.0.1:{self.port}"
        self.log = tmp_path / "master.log"
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        binary = Path(mori.__file__).parent / "umbp_master"
        if not binary.exists():
            pytest.skip("umbp_master is not installed")
        env = dict(os.environ, UMBP_HEARTBEAT_TTL_SEC=str(_HEARTBEAT_TTL_S))
        with open(self.log, "a") as log:
            self.process = subprocess.Popen(
                [str(binary), f"0.0.0.0:{self.port}", str(_free_port())],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.1)
        raise RuntimeError(f"umbp_master did not listen; see {self.log}")

    def kill(self) -> None:
        if self.process is not None:
            self.process.send_signal(signal.Signals["SIGCONT"])
            self.process.kill()
            self.process.wait(10)
            self.process = None

    def freeze(self) -> None:
        """Keep the sockets open but stop answering, like a hung master."""
        assert self.process is not None
        self.process.send_signal(signal.Signals["SIGSTOP"])

    def thaw(self) -> None:
        assert self.process is not None
        self.process.send_signal(signal.Signals["SIGCONT"])


@pytest.fixture
def master(tmp_path):
    server = Master(tmp_path)
    server.start()
    try:
        yield server
    finally:
        server.kill()


def _options(master_address: str, peer_base: int) -> dict:
    return {
        "master_address": master_address,
        "node_address": "127.0.0.1",
        "peer_service_port": peer_base,
        "capacity_bytes": 256 * 1024 * 1024,
        "dram_page_size": _LAYER_BYTES,
        "ranged_scratch_size": 16 * 1024 * 1024,
        "cache_remote_fetches": False,
        "ranged_locality_prefetch": False,
        "num_workers": 2,
        "timeout_ms": 60000,
    }


def _runtime(options: dict) -> DistributedRuntime:
    return DistributedRuntime(UMBPRuntimeConfig("distributed", options))


def _pattern(device: str) -> torch.Tensor:
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


def _producer(options: dict, stored, release) -> None:
    torch.accelerator.set_device_index(0)
    worker = _runtime(options).create_worker_handle("ns", RankTopology(), None)
    source = _pattern("cuda:0")
    worker.register_buffers({"kv": source})
    job = worker.wait(worker.store(_plans(source)))
    assert job.status is TransferJobStatus.COMPLETED, job.error
    worker.publish(job)
    stored.put(True)
    # The objects live in this node's pool, so it must outlive the readers.
    release.get(timeout=600)
    worker.close()


class Producer:
    def __init__(self, options: dict) -> None:
        context = mp.get_context("spawn")
        self._stored = context.Queue()
        self._release = context.Queue()
        self.process = context.Process(
            target=_producer, args=(options, self._stored, self._release)
        )
        self.process.start()
        assert self._stored.get(timeout=180)

    def close(self) -> None:
        if self.process.is_alive():
            self._release.put(True)
            self.process.join(60)
        assert self.process.exitcode == 0


@pytest.fixture
def producer(master):
    peer_base = _free_port(_PEER_PORT_SPAN)
    running = Producer(_options(master.address, peer_base))
    try:
        yield running, peer_base
    finally:
        running.close()


def _consumer(master: Master, peer_base: int):
    torch.accelerator.set_device_index(1)
    runtime = _runtime(_options(master.address, peer_base))
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    destination = torch.zeros(2, 4, _LAYER_BYTES, dtype=torch.uint8, device="cuda:1")
    worker.register_buffers({"kv": destination})
    return scheduler, worker, destination


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return predicate()


def test_restore_from_another_node_on_another_gpu(master, producer):
    _, peer_base = producer
    scheduler, worker, destination = _consumer(master, peer_base)
    assert list(scheduler.lookup([*_KEYS, "never-stored"])) == [True, True, False]

    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is TransferJobStatus.COMPLETED, loaded.error
    torch.accelerator.synchronize()

    assert torch.equal(destination[:, 2:4], _pattern("cuda:1"))
    assert torch.count_nonzero(destination[:, 0:2]) == 0
    worker.close()
    scheduler.close()


def test_objects_leave_the_index_when_their_node_exits(master, producer):
    running, peer_base = producer
    scheduler, worker, _ = _consumer(master, peer_base)
    assert list(scheduler.lookup(_KEYS)) == [True, True]

    running.close()

    assert _wait_for(lambda: list(scheduler.lookup(_KEYS)) == [False, False], 10)
    worker.close()
    scheduler.close()


def test_master_loss_degrades_to_misses_and_failed_transfers(master, producer):
    _, peer_base = producer
    scheduler, worker, destination = _consumer(master, peer_base)
    assert list(scheduler.lookup(_KEYS)) == [True, True]

    master.kill()

    assert list(scheduler.lookup(_KEYS)) == [False, False]
    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is not TransferJobStatus.COMPLETED
    stored = worker.wait(worker.store(_plans(destination)))
    assert stored.status is not TransferJobStatus.COMPLETED
    assert scheduler.clear() is False
    worker.close()
    scheduler.close()


def test_restarted_master_relearns_every_live_node(master, producer):
    _, peer_base = producer
    scheduler, worker, destination = _consumer(master, peer_base)

    master.kill()
    master.start()

    # Nodes re-register on their next heartbeat and resend a full snapshot.
    assert _wait_for(
        lambda: list(scheduler.lookup(_KEYS)) == [True, True], 6 * _HEARTBEAT_TTL_S
    )
    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is TransferJobStatus.COMPLETED, loaded.error
    torch.accelerator.synchronize()
    assert torch.equal(destination[:, 2:4], _pattern("cuda:1"))
    worker.close()
    scheduler.close()


def test_hung_master_costs_one_bounded_lookup(master, producer):
    _, peer_base = producer
    torch.accelerator.set_device_index(1)
    options = dict(_options(master.address, peer_base), lookup_timeout_ms=500)
    scheduler = _runtime(options).create_scheduler_handle("ns", RankTopology(), None)
    assert list(scheduler.lookup(_KEYS)) == [True, True]

    master.freeze()
    try:
        started = time.monotonic()
        assert list(scheduler.lookup(_KEYS)) == [False, False]
        first = time.monotonic() - started
        started = time.monotonic()
        assert list(scheduler.lookup(_KEYS)) == [False, False]
        second = time.monotonic() - started
    finally:
        master.thaw()

    assert first < 2
    assert second < 0.1
    assert _wait_for(
        lambda: list(scheduler.lookup(_KEYS)) == [True, True], 6 * _HEARTBEAT_TTL_S
    )
    scheduler.close()


def test_lookup_client_allocates_no_pool(master):
    def rss() -> int:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * mmap.PAGESIZE

    before = rss()
    # Pools are prefaulted by default, so a 256 MiB pool would be resident.
    scheduler = _runtime(
        _options(master.address, _free_port(_PEER_PORT_SPAN))
    ).create_scheduler_handle("ns", RankTopology(), None)

    assert rss() - before < 64 * 1024 * 1024
    assert list(scheduler.lookup(["absent"])) == [False]
    scheduler.close()
