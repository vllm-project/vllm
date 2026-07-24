# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import msgspec
import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPAsyncMPClient


def _make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req-1",
        external_req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # This unit test does not initialize distributed state and is CPU-only.
    return False


def test_dp_client_first_req_notifies_engine_index():
    client = object.__new__(DPAsyncMPClient)
    client.current_wave = 0
    client.client_index = 0
    client.engines_running = False
    client.core_engines = [b"\x00\x00", b"\x01\x00"]

    class _FakeSocket:

        def __init__(self):
            self.sent = []

        async def send(self, msg):
            self.sent.append(msg)

    client.first_req_send_socket = _FakeSocket()

    async def _fake_send_input(_request_type, _request, _engine=None):
        return None

    client._send_input = _fake_send_input
    client._ensure_stats_update_task = lambda: None
    client._ensure_output_queue_task = lambda: None
    client.get_core_engine_for_request = lambda _request: client.core_engines[1]

    request = _make_request()
    asyncio.run(client.add_request_async(request))

    assert len(client.first_req_send_socket.sent) == 1
    payload = msgspec.msgpack.decode(client.first_req_send_socket.sent[0])
    assert payload[0] == "FIRST_REQ"
    assert payload[1] == 1
