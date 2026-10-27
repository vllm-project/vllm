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
from vllm.v1.token_buffer import (
    TokenBuffer,
    TokenIdsView,
    extend_from_slice,
    rfind_tokens,
)


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
    frames = MsgpackEncoder().encode(engine_request)
    decoded = MsgpackDecoder(EngineCoreRequest).decode(frames)
    # Decoded without a copy, as a view of the received frame.
    assert isinstance(decoded.prompt_token_ids, memoryview)
    assert decoded.prompt_token_ids.obj is frames[0]
    request = Request.from_engine_core_request(decoded, block_hasher=None)
    assert request.prompt_token_ids == ids and request.num_prompt_tokens == 3
    assert type(request.prompt_token_ids[0]) is int
    request.append_output_token_ids([5, 6])
    assert request.all_token_ids[-3:] == [129279, 5, 6]
    assert request.prompt_token_ids == ids


def test_mp_client_sends_prompt_as_int32_bytes():
    """The Python frontend's engine-core client packs the prompt for the wire
    and leaves the request it keeps for output processing unchanged."""
    from vllm.v1.engine.core_client import _with_packed_prompt

    ids = [1, 70000, 129279]
    request = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    wire = _with_packed_prompt(request)
    assert request.prompt_token_ids is ids
    decoded = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(wire))
    assert isinstance(decoded.prompt_token_ids, memoryview)
    engine_request = Request.from_engine_core_request(decoded, block_hasher=None)
    assert engine_request.prompt_token_ids == ids


def test_request_token_ids_hold_no_int_references():
    """Long prompts must not add per-token objects for the cyclic GC to walk."""
    request = Request("r", list(range(1024)), SamplingParams(max_tokens=4), None)
    for tokens in (request.prompt_token_ids, request._all_token_ids):
        assert not any(isinstance(r, int) for r in gc.get_referents(tokens))
    assert request.all_token_ids.copy() == list(range(1024))


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_token_buffer_from_packed_bytes(wrap):
    ids = [1, 70000, 129279]
    tokens = TokenBuffer(wrap(struct.pack(f"<{len(ids)}i", *ids)))
    assert type(tokens) is TokenBuffer and tokens == ids
    with pytest.raises(ValueError):
        TokenBuffer(wrap(b"\x01\x02\x03"))


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_token_buffer_pickles(protocol):
    tokens = TokenBuffer([1, 70000, 129279])
    restored = pickle.loads(pickle.dumps(tokens, protocol=protocol))
    assert type(restored) is TokenBuffer and restored == [1, 70000, 129279]


@pytest.mark.parametrize("length", [5, 3000])
def test_rfind_tokens_buffer_matches_list(length):
    """The numpy search of a TokenBuffer, over chunks growing back from the end,
    finds the same index as the Python scan of a list."""
    ids = [(i * 7) % 11 for i in range(length)]
    tokens = TokenBuffer(ids)
    for targets in [(3,), (3, 5), (99,), (None, 4)]:
        for start, stop in [(0, None), (2, None), (0, length - 1), (length, None)]:
            expected = rfind_tokens(ids, targets, start, stop)
            assert rfind_tokens(tokens, targets, start, stop) == expected
            hits = [i for i in range(start, stop or length) if ids[i] in targets]
            assert expected == (hits[-1] if hits else -1)


def test_extend_from_slice():
    src = TokenBuffer(range(10))
    for dst in (TokenBuffer([7]), [7]):
        extend_from_slice(dst, src, 2, 5)
        assert dst == [7, 2, 3, 4]
    extend_from_slice(dst, list(range(10)), 8, 8)
    assert dst == [7, 2, 3, 4]


def test_token_ids_view_searches_in_place():
    tokens = TokenBuffer([1, 2, 3, 4])
    view = TokenIdsView(tokens, [5, 6])
    assert len(view) == 6 and rfind_tokens(view, (2, 6)) == 5
    view.pop()
    view.pop()
    view.pop()
    # Now tokens[:3]: the search stops before the popped 4 and the suffix.
    assert len(view) == 3 and rfind_tokens(view, (4, 6)) == -1
    assert rfind_tokens(view, (2,)) == 1
    assert view._list is None  # searching didn't copy the tokens
    assert list(view) == [1, 2, 3] and view[-1] == 3
    view.pop()
    assert list(view) == [1, 2] and rfind_tokens(view, (3,)) == -1


def test_prefill_token_ids_shares_prompt_until_outputs():
    request = Request("r", [1, 2, 3], SamplingParams(max_tokens=4), None)
    # Same object, so NewRequestData pickles the prompt once.
    assert request.prefill_token_ids is request.prompt_token_ids
    request.append_output_token_ids(4)
    assert request.prefill_token_ids is request._all_token_ids
    assert request.prefill_token_ids == [1, 2, 3, 4]


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
