# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import zmq

from vllm.config import DeviceConfig, ParallelConfig, SnapshotConfig, VllmConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine.core_client import AsyncMPClient, MPClient


@pytest.mark.parametrize(
    "snapshot_config,enable_elastic_ep,expected_handover",
    [
        (None, False, False),
        (SnapshotConfig(), False, True),
        (None, True, True),
    ],
)
def test_router_handover_feature_gates(
    snapshot_config,
    enable_elastic_ep,
    expected_handover,
    tmp_path,
):
    handover = []
    input_address = f"ipc://{tmp_path}/input"
    output_address = f"ipc://{tmp_path}/output"
    done = Event()

    def ready_peer():
        with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.IDENTITY, b"\0\0")
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect(input_address)
            assert socket.poll(10000, zmq.POLLOUT), "client did not bind its input socket"
            socket.send(b"")
            assert done.wait(10), "client did not complete its handshake"

    def make_socket(ctx, address, socket_type, **kwargs):
        if socket_type == zmq.ROUTER:
            handover.append(kwargs["router_handover"])
        return make_zmq_socket(ctx, address, socket_type, **kwargs)

    config = VllmConfig(
        parallel_config=ParallelConfig(
            enable_elastic_ep=enable_elastic_ep,
            enable_eplb=enable_elastic_ep,
            eplb_config={"use_async": False},
        ),
        device_config=DeviceConfig(device="cpu"),
        snapshot_config=snapshot_config,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        peer = pool.submit(ready_peer)
        try:
            with patch("vllm.v1.engine.core_client.make_zmq_socket", side_effect=make_socket):
                client = MPClient(
                    asyncio_mode=False,
                    vllm_config=config,
                    executor_class=object,
                    collect_stats=False,
                    client_addresses={"input_address": input_address, "output_address": output_address},
                )
                try:
                    assert (client.snapshot_monitor is not None) is (snapshot_config is not None)
                finally:
                    client.shutdown()
        finally:
            done.set()
        peer.result(timeout=10)
    assert handover == [expected_handover]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_tcp_input_transport,expected_order",
    [
        (True, ["ready", "resume"]),
        (False, ["resume", "ready"]),
    ],
)
async def test_resume_engine_order_matches_transport(
    is_tcp_input_transport,
    expected_order,
):
    calls = []

    async def wait_for_engines_ready():
        calls.append("ready")

    async def call_utility_async(method, data_parallel_master_ip, model_path):
        assert method == "resume"
        assert data_parallel_master_ip == "10.0.0.2"
        assert model_path == "/snapshot/model"
        calls.append("resume")

    client = SimpleNamespace(
        _is_tcp_input_transport=is_tcp_input_transport,
        wait_for_engines_ready=wait_for_engines_ready,
        call_utility_async=call_utility_async,
    )

    await AsyncMPClient._resume_engines(
        client,
        "10.0.0.2",
        "/snapshot/model",
    )

    assert calls == expected_order


@pytest.mark.asyncio
async def test_snapshot_lifecycle_requires_snapshot_config():
    client = SimpleNamespace(snapshot_monitor=None)

    with pytest.raises(RuntimeError, match="--snapshot-config"):
        await AsyncMPClient.suspend_async(client, "/snapshot/model")
