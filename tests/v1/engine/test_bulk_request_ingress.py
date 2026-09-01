# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import SyncMPClient


def make_client() -> SyncMPClient:
    client = object.__new__(SyncMPClient)
    client.is_dp = False
    client.engines_running = False
    client._send_input = MagicMock()
    return client


@pytest.mark.parametrize("request_count", [1, 3])
def test_sync_mp_client_add_requests_uses_one_message(request_count: int):
    client = make_client()
    requests = [object() for _ in range(request_count)]

    client.add_requests(requests)

    client._send_input.assert_called_once_with(
        EngineCoreRequestType.ADD_BATCH, requests
    )


def test_sync_mp_client_add_requests_empty_is_noop():
    client = make_client()

    client.add_requests([])

    client._send_input.assert_not_called()


def test_sync_mp_client_single_add_request_unchanged():
    client = make_client()
    request = object()

    client.add_request(request)

    client._send_input.assert_called_once_with(EngineCoreRequestType.ADD, request)


def test_sync_mp_client_add_requests_preserves_send_failure():
    client = make_client()
    client._send_input.side_effect = RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        client.add_requests([object()])


def test_engine_core_handles_complete_batch_before_returning():
    core = object.__new__(EngineCoreProc)
    core._reject_add_in_shutdown = MagicMock(return_value=False)
    core.add_request = MagicMock(side_effect=[None, RuntimeError("bad request"), None])
    core._send_error_outputs_to_client = MagicMock()
    requests = [
        (SimpleNamespace(request_id=f"req-{i}", client_index=0), i) for i in range(3)
    ]

    core._handle_client_request(EngineCoreRequestType.ADD_BATCH, requests)

    assert core.add_request.call_args_list == [
        call(request, wave) for request, wave in requests
    ]
    core._send_error_outputs_to_client.assert_called_once_with(["req-1"], 0)
