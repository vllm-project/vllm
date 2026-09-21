# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus


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


def _engine_request_with_salt(cache_salt: str | None) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=cache_salt,
        data_parallel_rank=None,
    )


def test_request_accepts_omitted_and_valid_cache_salt():
    omitted = Request.from_engine_core_request(
        _engine_request_with_salt(None), block_hasher=None
    )
    assert omitted.cache_salt is None

    valid = Request.from_engine_core_request(
        _engine_request_with_salt("a" * 128), block_hasher=None
    )
    assert valid.cache_salt == "a" * 128


def test_request_rejects_empty_cache_salt():
    with pytest.raises(ValueError, match="non-empty string"):
        Request.from_engine_core_request(
            _engine_request_with_salt(""), block_hasher=None
        )


def test_request_rejects_oversized_cache_salt():
    with pytest.raises(ValueError, match="at most 128 characters"):
        Request.from_engine_core_request(
            _engine_request_with_salt("B" * 129), block_hasher=None
        )


@pytest.mark.parametrize("cache_salt", ["tenant@victim", "../../x", r"..\x", "a\0b"])
def test_request_rejects_forbidden_cache_salt_characters(cache_salt: str):
    with pytest.raises(ValueError, match="must not contain"):
        Request.from_engine_core_request(
            _engine_request_with_salt(cache_salt), block_hasher=None
        )
