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

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list

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
