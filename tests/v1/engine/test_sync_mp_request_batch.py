# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, Mock, call

import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def make_client() -> SyncMPClient:
    client = object.__new__(SyncMPClient)
    client.is_dp = False
    client._pending_add_batch = None
    client._send_input = Mock()
    return client


def make_request(request_id: str) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_batches_add_requests():
    client = make_client()
    requests = [MagicMock(spec=EngineCoreRequest) for _ in range(2)]

    with client.batch_add_requests():
        for request in requests:
            client.add_request(request)

    client._send_input.assert_called_once_with(
        EngineCoreRequestType.ADD_BATCH, requests
    )


def test_preserves_single_add():
    client = make_client()
    request = MagicMock(spec=EngineCoreRequest)

    with client.batch_add_requests():
        client.add_request(request)

    client._send_input.assert_called_once_with(EngineCoreRequestType.ADD, request)


def test_preserves_individual_adds_for_data_parallel():
    client = make_client()
    client.is_dp = True
    requests = [MagicMock(spec=EngineCoreRequest) for _ in range(2)]

    with client.batch_add_requests():
        for request in requests:
            client.add_request(request)

    assert client._send_input.call_args_list == [
        call(EngineCoreRequestType.ADD, requests[0]),
        call(EngineCoreRequestType.ADD, requests[1]),
    ]
    assert client._pending_add_batch is None


def test_discards_failed_batch():
    client = make_client()

    with pytest.raises(RuntimeError), client.batch_add_requests():
        client.add_request(MagicMock(spec=EngineCoreRequest))
        raise RuntimeError("frontend preprocessing failed")

    client._send_input.assert_not_called()
    assert client._pending_add_batch is None


def test_request_batch_msgpack_round_trip():
    requests = [make_request("0"), make_request("1")]

    encoded = MsgpackEncoder().encode(requests)
    decoded = MsgpackDecoder(list[EngineCoreRequest]).decode(encoded)

    assert [request.request_id for request in decoded] == ["0", "1"]
    assert all(request.sampling_params.max_tokens == 1 for request in decoded)


def test_engine_core_admits_entire_batch():
    core = object.__new__(EngineCoreProc)
    core._reject_add_in_shutdown = Mock(return_value=False)
    core.add_request = Mock()
    requests = [(MagicMock(), 3), (MagicMock(), 3)]

    core._handle_client_request(EngineCoreRequestType.ADD_BATCH, requests)

    assert core.add_request.call_args_list == [
        call(requests[0][0], 3),
        call(requests[1][0], 3),
    ]
