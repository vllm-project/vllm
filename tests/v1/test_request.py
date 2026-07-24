# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
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


def test_cache_checkpoint_boundaries_are_normalized_from_engine_request():
    core_request = EngineCoreRequest(
        request_id="request",
        prompt_token_ids=list(range(100)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        cache_checkpoint_boundaries=(48, 0, 48, 100, 96),
        cache_checkpoint_decode_end=True,
    )

    request = Request.from_engine_core_request(core_request, block_hasher=None)

    assert request.cache_checkpoint_boundaries == (48, 96)
    assert request.cache_checkpoint_decode_end is True
