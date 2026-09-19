# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
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
def test_request_priority_comparison():
    """Test that Request.__lt__ orders by:
    1. priority (lower number = higher priority)
    2. num_preemptions (higher count = higher priority)
    3. arrival_time (earlier arrival = higher priority)
    4. request_id (tiebreaker)
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    def make_req(
        req_id: str, priority: int, preemptions: int, arrival: float
    ) -> Request:
        req = Request(
            request_id=req_id,
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=5),
            pooling_params=None,
            priority=priority,
            arrival_time=arrival,
        )
        req.num_preemptions = preemptions
        return req

    # 1. Higher priority (lower numerical value) wins
    r_high = make_req("high", priority=0, preemptions=0, arrival=10.0)
    r_low = make_req("low", priority=1, preemptions=5, arrival=1.0)
    assert r_high < r_low

    # 2. Equal priority: higher num_preemptions wins (even if arrival is later)
    r_preempted = make_req("preempted", priority=0, preemptions=1, arrival=10.0)
    r_unstarted = make_req("unstarted", priority=0, preemptions=0, arrival=1.0)
    assert r_preempted < r_unstarted

    # 3. Equal priority & equal preemptions: earlier arrival_time wins
    r_early = make_req("early", priority=0, preemptions=1, arrival=2.0)
    r_late = make_req("late", priority=0, preemptions=1, arrival=5.0)
    assert r_early < r_late

    # 4. Equal priority, preemptions, arrival_time: request_id tiebreaker
    r_id_a = make_req("req_a", priority=0, preemptions=0, arrival=1.0)
    r_id_b = make_req("req_b", priority=0, preemptions=0, arrival=1.0)
    assert r_id_a < r_id_b

    # 5. Starvation cap: preemptions capped at 3
    r_cap_3 = make_req("cap_3", priority=0, preemptions=3, arrival=1.0)
    r_cap_10 = make_req("cap_10", priority=0, preemptions=10, arrival=1.0)
    assert r_cap_3.sort_key[1] == -3
    assert r_cap_10.sort_key[1] == -3
    # With equal capped preemptions, earlier arrival_time breaks the tie
    r_cap_10_late = make_req("cap_10_late", priority=0, preemptions=10, arrival=2.0)
    assert r_cap_3 < r_cap_10_late
