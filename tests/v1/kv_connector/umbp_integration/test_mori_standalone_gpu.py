# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StandaloneRuntime against a real umbp_standalone_server on ROCm GPUs."""

import multiprocessing as mp
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import vllm.envs as envs
from tests.v1.kv_connector.rocm_pd_accuracy_utils import _REVISIONS, QWEN_MODEL
from tests.v1.kv_connector.umbp_integration.test_mori_token_e2e import (
    BLOCK_SIZE,
    PARAMS,
    _assert_same_outputs,
    _external_tokens,
    _prompt,
    _reset_gpu_prefix_cache,
)
from tests.v1.kv_connector.umbp_test_utils import run_in_new_process
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

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="UMBP requires ROCm"
)

_LAYER_BYTES = 1 << 20
_KEYS = ("standalone-block-0", "standalone-block-1")


class _Server:
    def __init__(self, address: str, process: subprocess.Popen):
        self.address = address
        self.process = process

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(10)


@pytest.fixture
def server(tmp_path, socket_dir):
    import mori

    binary = Path(mori.__file__).with_name("umbp_standalone_server")
    assert binary.is_file(), f"MORI was built without UMBP: no {binary}"
    address = f"unix://{socket_dir}/umbp.grpc.sock"
    env = dict(
        os.environ,
        UMBP_STANDALONE_ADDRESS=address,
        UMBP_DRAM_CAPACITY=str(256 * 1024 * 1024),
    )
    with open(tmp_path / "server.log", "w") as log:
        process = subprocess.Popen(
            [binary], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            yield _Server(address, process)
        finally:
            process.send_signal(signal.SIGCONT)
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)


def _runtime(address: str, **options) -> StandaloneRuntime:
    return StandaloneRuntime(
        UMBPRuntimeConfig(
            "standalone",
            {"endpoint": address, "startup_timeout_ms": 20000, **options},
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


def _produce(address: str) -> None:
    torch.accelerator.set_device_index(0)
    worker = _runtime(address).create_worker_handle("ns", RankTopology(), None)
    try:
        source = _pattern("cuda:0")
        worker.register_buffers({"kv": source})
        stored = worker.wait(worker.store(_plans(source)))
        assert stored.status is TransferJobStatus.COMPLETED, stored.error
        worker.publish(stored)
    finally:
        worker.close()


_OBJECTS = 16


def _fill(writer: int, block: int) -> int:
    return writer * 64 + block + 1


def _block_plans(cache: torch.Tensor, prefix: str) -> list[BlockTransferPlan]:
    return [
        BlockTransferPlan(
            f"{prefix}-{block}",
            block,
            ranges=(
                KVRange(
                    "kv",
                    0,
                    block,
                    cache[block].data_ptr(),
                    _LAYER_BYTES,
                    _LAYER_BYTES,
                    0,
                ),
            ),
        )
        for block in range(_OBJECTS)
    ]


def _store_concurrently(address: str, writer: int, start) -> None:
    torch.accelerator.set_device_index(0)
    worker = _runtime(address).create_worker_handle("ns", RankTopology(), None)
    try:
        own = torch.empty(_OBJECTS, _LAYER_BYTES, dtype=torch.uint8, device="cuda")
        shared = torch.empty_like(own)
        for block in range(_OBJECTS):
            own[block].fill_(_fill(writer, block))
            shared[block].fill_(_fill(writer, _OBJECTS + block))
        worker.register_buffers({"own": own, "shared": shared})
        plans = _block_plans(own, f"writer{writer}") + _block_plans(shared, "shared")
        assert start.wait(60)
        stored = worker.wait(worker.store(plans))
        assert stored.status is TransferJobStatus.COMPLETED, stored.error
        worker.publish(stored)
    finally:
        worker.close()


def _close_while_server_stopped(address: str, server_pid: int, scenario: str):
    torch.accelerator.set_device_index(0)
    source = torch.arange(_LAYER_BYTES, dtype=torch.uint8, device="cuda:0")
    kv_range = KVRange("kv", 0, 0, source.data_ptr(), _LAYER_BYTES, _LAYER_BYTES, 0)
    plan = BlockTransferPlan("stuck", 0, ranges=(kv_range,))
    if scenario == "load":
        writer = _runtime(address).create_worker_handle("ns", RankTopology(), None)
        writer.register_buffers({"kv": source})
        stored = writer.wait(writer.store([plan]))
        assert stored.status is TransferJobStatus.COMPLETED, stored.error
        writer.publish(stored)
        writer.close()

    runtime = _runtime(address, timeout_ms=200, lookup_timeout_ms=200)
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    worker.register_buffers({"kv": source})
    os.kill(server_pid, signal.SIGSTOP)
    if scenario in ("store", "load"):
        job = getattr(worker, scenario)([plan])
        time.sleep(0.5)
        assert worker.poll(job) is None
    elif scenario == "lookup":
        started = time.monotonic()
        assert list(scheduler.lookup(["stuck"])) == [None]
        assert list(scheduler.lookup(["stuck"])) == [None]
        assert time.monotonic() - started < 5
    scheduler.close()
    worker.close()


def test_engine_restores_after_writer_exits(server, vllm_runner, monkeypatch):
    """A fresh engine restores the writer's KV and reproduces its outputs."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    runner_kwargs = dict(
        model_name=QWEN_MODEL,
        revision=_REVISIONS[QWEN_MODEL],
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        disable_log_stats=False,
        gpu_memory_utilization=0.1,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        kv_transfer_config={
            "kv_connector": "UMBPStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "mode": "standalone",
                "endpoint": server.address,
                "key_namespace": "engine-restart",
                "startup_timeout_ms": 20000,
            },
        },
    )
    prompts = [
        _prompt(seed, 4 * BLOCK_SIZE + tail)
        for seed, tail in enumerate((1, 7, BLOCK_SIZE - 1))
    ]
    with vllm_runner(**runner_kwargs) as writer:
        computed = writer.llm.generate(prompts, PARAMS, use_tqdm=False)
        assert _external_tokens(writer.llm) == 0
        _reset_gpu_prefix_cache(writer.llm)

    with vllm_runner(**runner_kwargs) as reader:
        restored = reader.llm.generate(prompts, PARAMS, use_tqdm=False)
        assert _external_tokens(reader.llm) == 3 * 4 * BLOCK_SIZE
        _assert_same_outputs(computed, restored)


@pytest.mark.skipif(current_platform.device_count() < 2, reason="needs 2 GPUs")
def test_objects_outlive_the_writer_and_restore_on_another_gpu(server):
    run_in_new_process(_produce, server.address)

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

    assert torch.equal(destination[:, 2:4], _pattern("cuda:1"))
    assert torch.count_nonzero(destination[:, 0:2]) == 0
    worker.close()
    scheduler.close()


def test_clear_removes_objects_for_every_client(server):
    run_in_new_process(_produce, server.address)
    scheduler = _runtime(server.address).create_scheduler_handle(
        "ns", RankTopology(), None
    )
    assert list(scheduler.lookup(_KEYS)) == [True, True]

    assert scheduler.clear() is True

    assert list(scheduler.lookup(_KEYS)) == [False, False]
    scheduler.close()


def test_server_loss_degrades_to_misses_and_failed_transfers(server):
    run_in_new_process(_produce, server.address)
    runtime = _runtime(server.address)
    scheduler = runtime.create_scheduler_handle("ns", RankTopology(), None)
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    destination = torch.zeros(2, 4, _LAYER_BYTES, dtype=torch.uint8, device="cuda")
    worker.register_buffers({"kv": destination})

    server.kill()

    assert list(scheduler.lookup(_KEYS)) == [False, False]
    loaded = worker.wait(worker.load(_plans(destination, block_offset=2)))
    assert loaded.status is TransferJobStatus.FAILED
    stored = worker.wait(worker.store(_plans(destination)))
    assert stored.status is TransferJobStatus.FAILED
    assert scheduler.clear() is False
    worker.close()
    scheduler.close()


@pytest.mark.parametrize("scenario", ["store", "load", "lookup", "idle"])
def test_process_exits_while_the_server_stays_stopped(server, scenario):
    """A server that never resumes must not hold up close() or process exit."""
    run_in_new_process(
        _close_while_server_stopped,
        server.address,
        server.process.pid,
        scenario,
        timeout=180,
    )


def test_concurrent_writers_store_whole_objects(server):
    """Clients storing at once, even to the same keys, never tear an object."""
    start = mp.get_context("spawn").Event()
    with ThreadPoolExecutor(2) as pool:
        writers = [
            pool.submit(
                run_in_new_process, _store_concurrently, server.address, w, start
            )
            for w in range(2)
        ]
        start.set()
        for writer in writers:
            writer.result()

    runtime = _runtime(server.address)
    worker = runtime.create_worker_handle("ns", RankTopology(), None)
    restored = torch.zeros(3, _OBJECTS, _LAYER_BYTES, dtype=torch.uint8, device="cuda")
    worker.register_buffers({"kv": restored})
    plans = [
        *_block_plans(restored[0], "writer0"),
        *_block_plans(restored[1], "writer1"),
        *_block_plans(restored[2], "shared"),
    ]
    loaded = worker.wait(worker.load(plans))
    assert loaded.status is TransferJobStatus.COMPLETED, loaded.error
    torch.accelerator.synchronize()
    worker.close()

    for writer in range(2):
        for block in range(_OBJECTS):
            assert bool((restored[writer, block] == _fill(writer, block)).all())
    for block in range(_OBJECTS):
        first = restored[2, block, 0].item()
        assert first in {_fill(w, _OBJECTS + block) for w in range(2)}
        assert bool((restored[2, block] == first).all())
