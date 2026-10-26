# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import pickle
import struct

import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.token_buffer import TokenBuffer


def test_request_decodes_int32_token_bytes_from_rust_frontend():
    """vllm-rs sends prompt ids as little-endian int32 bytes."""
    ids = [1, 70000, 129279]
    engine_request = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=struct.pack(f"<{len(ids)}i", *ids),  # type: ignore[arg-type]
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    decoded = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(engine_request)
    )
    request = Request.from_engine_core_request(decoded, block_hasher=None)
    assert request.prompt_token_ids == ids and request.num_prompt_tokens == 3
    assert type(request.prompt_token_ids[0]) is int
    request.append_output_token_ids([5, 6])
    assert request.all_token_ids[-3:] == [129279, 5, 6]
    assert request.prompt_token_ids == ids


def test_request_token_ids_hold_no_int_references():
    """Long prompts must not add per-token objects for the cyclic GC to walk."""
    request = Request("r", list(range(1024)), SamplingParams(max_tokens=4), None)
    for tokens in (request.prompt_token_ids, request._all_token_ids):
        assert not any(isinstance(r, int) for r in gc.get_referents(tokens))
    assert request.all_token_ids.copy() == list(range(1024))


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_token_buffer_pickles(protocol):
    tokens = TokenBuffer([1, 70000, 129279])
    restored = pickle.loads(pickle.dumps(tokens, protocol=protocol))
    assert type(restored) is TokenBuffer and restored == [1, 70000, 129279]


def test_token_buffer_pickles_out_of_band_snapshot():
    """With protocol 5 the ids go out of band, as a snapshot that can outlive the
    pickling (zmq sends without copying) while the buffer keeps growing."""
    tokens = TokenBuffer(range(1000))
    oob: list[pickle.PickleBuffer] = []
    data = pickle.dumps(tokens, protocol=5, buffer_callback=oob.append)
    assert len(oob) == 1 and len(data) < 200
    held = oob[0].raw()
    tokens.append(1000)  # Would raise BufferError if `held` were a view of tokens.
    assert len(held) == 4000
    restored = pickle.loads(data, buffers=[bytearray(held)])
    assert type(restored) is TokenBuffer and restored == list(range(1000))


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    """Test that Request preserves session ID from EngineCoreRequest."""
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_request_copies_kv_hints_from_engine_core_request():
    """Test that Request preserves KV hints from EngineCoreRequest."""
    kv_hints = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="msg-1",
        actions=[
            KvHintAction(
                action_id="action-1",
                action_type="example.action",
                action_version="1.0",
                payload={},
            )
        ],
    )
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        kv_hints=kv_hints,
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.kv_hints == kv_hints
