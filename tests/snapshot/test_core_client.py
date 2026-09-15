# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import zmq

from vllm.config import SnapshotConfig
from vllm.v1.engine.core_client import AsyncMPClient, MPClient


class _Socket:
    def close(self, *, linger=0):
        pass


class _ReadySocket:
    def poll(self, timeout):
        return 1

    def recv_multipart(self):
        return b"\0\0", b""


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
):
    handover = []

    def make_socket(_ctx, _address, socket_type, **kwargs):
        if socket_type == zmq.ROUTER:
            handover.append(kwargs["router_handover"])
        return _Socket()

    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_index=0,
        data_parallel_size_local=1,
        data_parallel_rank_local=None,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
        local_engines_only=False,
        enable_elastic_ep=enable_elastic_ep,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        snapshot_config=snapshot_config,
    )

    with (
        patch("vllm.v1.engine.core_client.make_zmq_socket", side_effect=make_socket),
        patch(
            "vllm.v1.engine.core_client.zmq.Socket.shadow",
            return_value=_ReadySocket(),
        ),
    ):
        client = MPClient(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=object,
            log_stats=False,
            client_addresses={
                "input_address": "ipc:///tmp/input",
                "output_address": "ipc:///tmp/output",
            },
        )
        assert (client.snapshot_monitor is not None) is (snapshot_config is not None)
        client.shutdown()

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
