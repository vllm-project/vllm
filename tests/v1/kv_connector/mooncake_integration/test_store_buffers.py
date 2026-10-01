# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Store buffer transfers, independent of model loading and hint semantics."""

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_transfer_manager
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.control import (
    MooncakeControlAdapter,
    MooncakeControlWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    KeyMetadata,
    RankLocalStoreLayout,
)
from vllm.platforms import current_platform
from vllm.utils.hashing import sha256
from vllm.utils.network_utils import get_open_ports_list
from vllm.v1.core.kv_cache_utils import get_block_hash, init_none_hash
from vllm.v1.kv_hints.control import ControlContext, OperationRegistry, OperationStatus
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.resources import KVResourceController
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex

store_module = pytest.importorskip("mooncake.store")
pytestmark = [
    pytest.mark.distributed,
    pytest.mark.skipif(
        not current_platform.is_cuda() or current_platform.device_count() < 2,
        reason="Requires two CUDA GPUs and the matching native Mooncake wheel",
    ),
]


@pytest.fixture(scope="module")
def store_master(tmp_path_factory):
    configured = os.environ.get("VLLM_TEST_MOONCAKE_MASTER")
    if configured:
        yield configured
        return
    binary = Path(sys.executable).parent / "mooncake_master"
    if not binary.exists():
        pytest.skip("Mooncake master executable is unavailable")
    port, metrics_port = get_open_ports_list(2)
    log_path = tmp_path_factory.mktemp("mooncake-store") / "master.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [
                str(binary),
                "--rpc_address=127.0.0.1",
                f"--rpc_port={port}",
                "--metrics_host=127.0.0.1",
                f"--metrics_port={metrics_port}",
                "--enable_metric_reporting=false",
                "--logtostderr=true",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                log.seek(0)
                pytest.fail(f"Mooncake master did not start:\n{log.read()}")
            yield f"127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture(scope="module")
def stores(store_master):
    clients = []
    try:
        for _ in range(2):
            client = store_module.MooncakeDistributedStore()
            clients.append(client)
            assert (
                client.setup(
                    "127.0.0.1",
                    "P2PHANDSHAKE",
                    128 << 20,
                    16 << 20,
                    "tcp",
                    "",
                    store_master,
                )
                == 0
            )
        yield clients
    finally:
        for client in reversed(clients):
            client.close()


@pytest.mark.parametrize("source_device", [0, 1])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("numel", [1024, 16384, 65536])
@pytest.mark.parametrize("buffers_per_key", [1, 3])
def test_store_gpu_buffers_preserve_values_across_a_partial_miss(
    stores, source_device, dtype, numel, buffers_per_key
):
    """A missing batch member must not corrupt successfully transferred GPU data."""
    destination_device = 1 - source_device
    producer, consumer = stores[source_device], stores[destination_device]
    sources = [
        [
            torch.arange(
                numel + 16 * i, device=f"cuda:{source_device}", dtype=torch.float32
            )
            .remainder(127)
            .add(group)
            .to(dtype)
            for i in range(buffers_per_key)
        ]
        for group in range(2)
    ]
    destinations = [
        [torch.zeros_like(t, device=f"cuda:{destination_device}") for t in group]
        for group in sources
    ]
    sentinel = torch.full(
        (256,), 17, device=f"cuda:{destination_device}", dtype=torch.uint8
    )
    torch.accelerator.synchronize(source_device)
    torch.accelerator.synchronize(destination_device)
    keys = [f"vllm-store-test-{uuid.uuid4().hex}" for _ in sources]
    registered = []
    try:
        for client, groups in (
            (producer, sources),
            (consumer, destinations + [[sentinel]]),
        ):
            for group in groups:
                for tensor in group:
                    assert client.register_buffer(tensor.data_ptr(), tensor.nbytes) == 0
                    registered.append((client, tensor))
        sizes = [[t.nbytes for t in group] for group in sources]
        assert producer.batch_put_from_multi_buffers(
            keys, [[t.data_ptr() for t in group] for group in sources], sizes
        ) == [0, 0]
        missing = f"missing-{uuid.uuid4().hex}"
        assert consumer.batch_is_exist(keys + [missing]) == [1, 1, 0]
        result = consumer.batch_get_into_multi_buffers(
            keys + [missing],
            [[t.data_ptr() for t in group] for group in destinations]
            + [[sentinel.data_ptr()]],
            sizes + [[sentinel.nbytes]],
        )
        # Native Get returns byte counts, unlike Put's zero success status.
        assert all(size >= 0 for size in result[:2]) and result[2] < 0
        torch.accelerator.synchronize(destination_device)
        for source_group, destination_group in zip(sources, destinations, strict=True):
            for source, destination in zip(
                source_group, destination_group, strict=True
            ):
                torch.testing.assert_close(
                    source.cpu(), destination.cpu(), rtol=0, atol=0
                )
        assert torch.all(sentinel.cpu() == 17)
    finally:
        for client, tensor in reversed(registered):
            client.unregister_buffer(tensor.data_ptr())
        for key in keys:
            producer.remove(key, True)


@pytest.mark.parametrize("outcome", ["success", "missing_rank", "reset"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_control_moves_real_gpu_buffers_before_publishing_hits(
    stores, outcome, dtype
):
    init_none_hash(sha256)
    manager, _, request, keys = make_transfer_manager()
    registry = OperationRegistry("buffer-engine", engine_epoch="buffer-epoch")
    index = SessionPrefixIndex(uuid.uuid4().hex, "buffer-epoch")
    declarations = RetentionDeclarations(index)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True, kv_connector_extra_config={}
        ),
    )
    adapter = MooncakeControlAdapter(config, manager.kv_cache_config, 4)
    executor = KVResourceController(manager, registry, declarations, (adapter,))
    controller = NativeKVController(registry, index, declarations, executor=executor)
    context = ControlContext("buffer-engine", "buffer-epoch", "native")
    target = {
        "kind": "hashes",
        "namespace": index.namespace,
        "blocks": [
            {
                "block_hash": request.block_hashes[0].hex(),
                "group_id": 0,
                "start_token": 0,
                "end_token": 4,
            }
        ],
    }

    def submit(action, extra):
        return controller.submit(
            KvHintsEnvelope(
                "0.1",
                action,
                [
                    KvHintAction("a", action, "1", {"target": target, **extra}),
                ],
            ),
            context,
        ).results[0]

    workers, tensors, layouts, registered_keys = [], [], [], []
    original_device = torch.accelerator.current_device_index()
    try:
        for rank, store in enumerate(stores):
            torch.accelerator.set_device_index(rank)
            tensor = torch.arange(64, device=f"cuda:{rank}", dtype=torch.float32)
            tensor = tensor.reshape(4, 16).add(100 * rank).to(dtype)
            assert store.register_buffer(tensor.data_ptr(), tensor.nbytes) == 0
            tensors.append(tensor)
            layout = RankLocalStoreLayout(
                KeyMetadata("control-test", rank, 0, 0, 0), 4, 4
            )
            layout.register_kv_caches([tensor], 4)
            layouts.append(layout)
            workers.append(
                MooncakeControlWorker(
                    store,
                    {0: layout},
                    rank=rank,
                    tp_rank=rank,
                    group_replication={0: 1},
                )
            )
        blocks = manager.allocate_slots(request, 4)
        assert blocks is not None
        manager.cache_blocks(request, 4)
        source_id = blocks.blocks[0][0].block_id
        expected = [tensor[source_id].cpu().clone() for tensor in tensors]
        manager.free(request)

        def drain(action):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                for worker in workers:
                    adapter.update(worker.poll())
                executor.progress()
                result = registry.get(action, "a")
                if result.status.terminal and not adapter.has_pending:
                    return result
                time.sleep(0.01)
            pytest.fail("Native control did not drain real Store transfers")

        result = submit("demote", {"destination": "mooncake"})
        assert result.status == OperationStatus.RUNNING
        jobs = adapter.take_metadata()
        assert len(jobs) == 1
        for rank, worker in enumerate(workers):
            torch.accelerator.set_device_index(rank)
            worker.submit(jobs, "store")
            key = layouts[rank].key_for(0, get_block_hash(keys[0]))
            registered_keys.append(f"kv-control:{jobs[0].scope}@{key}")
        result = drain("demote")
        assert result.status == OperationStatus.COMPLETED
        assert result.completed_steps == ("cold_copy_stored", "gpu_copies_reclaimed")
        assert manager.get_computed_blocks(request)[1] == 0
        for rank, tensor in enumerate(tensors):
            torch.accelerator.set_device_index(rank)
            tensor.zero_()
        if outcome == "missing_rank":
            stores[1].remove(registered_keys[1], True)
        assert (
            submit("prefetch", {"source": "mooncake"}).status == OperationStatus.RUNNING
        )
        jobs = adapter.take_metadata()
        assert len(jobs) == 1
        assert manager.get_computed_blocks(request)[1] == 0
        for rank, worker in enumerate(workers):
            torch.accelerator.set_device_index(rank)
            worker.submit(jobs, "load")
        if outcome == "reset":
            assert not manager.reset_prefix_cache()
        result = drain("prefetch")
        expected_status = {
            "success": OperationStatus.COMPLETED,
            "missing_rank": OperationStatus.MISSING,
            "reset": OperationStatus.CANCELLED,
        }[outcome]
        assert result.status == expected_status
        assert manager.get_computed_blocks(request)[1] == (
            4 if outcome == "success" else 0
        )
        assert not manager.get_transfer_manager().has_pending
        if outcome == "success":
            for rank, tensor in enumerate(tensors):
                torch.testing.assert_close(
                    tensor[jobs[0].block_id].cpu(),
                    expected[rank],
                    rtol=0,
                    atol=0,
                )
    finally:
        for worker in workers:
            worker.close()
        for store, tensor in zip(stores, tensors):
            store.unregister_buffer(tensor.data_ptr())
        for store, key in zip(stores, registered_keys):
            store.remove(key, True)
        torch.accelerator.set_device_index(original_device)
