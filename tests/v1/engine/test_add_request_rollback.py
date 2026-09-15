# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Roll back local request registration when engine add fails."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.serial_utils import MsgpackEncoder

pytestmark = pytest.mark.cpu_test

_OVERFLOW = 2**64


def _engine_request(
    request_id: str = "req-0",
    *,
    sampling_params: SamplingParams | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=sampling_params
        or SamplingParams(detokenize=False, max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_async_llm(add_request_async) -> AsyncLLM:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.output_handler = None
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    llm.output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    llm.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=None,
        max_num_queued_tokens=None,
    )
    llm.input_processor = SimpleNamespace(assign_request_id=lambda _req: None)
    llm.log_requests = False
    llm._run_output_handler = lambda: None
    llm.engine_core = SimpleNamespace(
        add_request_async=add_request_async,
        abort_requests_async=AsyncMock(),
        resources=SimpleNamespace(engine_dead=False),
        shutdown=MagicMock(),
    )
    return llm


def _make_llm_engine(add_request) -> LLMEngine:
    engine = LLMEngine.__new__(LLMEngine)
    engine.output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    engine.input_processor = SimpleNamespace(assign_request_id=lambda _req: None)
    engine.engine_core = SimpleNamespace(
        add_request=add_request,
        abort_requests=MagicMock(),
    )
    return engine


def test_async_add_request_rolls_back_on_encode_overflow():
    encoder = MsgpackEncoder()

    async def add_request_async(request: EngineCoreRequest) -> None:
        encoder.encode(request)

    llm = _make_async_llm(add_request_async)
    request = _engine_request()
    request.priority = _OVERFLOW
    queue = RequestOutputCollector(RequestOutputKind.CUMULATIVE, request.request_id)

    async def _run() -> None:
        with pytest.raises(OverflowError):
            await llm._add_request(request, None, None, 0, queue)

    asyncio.run(_run())

    assert llm.output_processor.request_states == {}
    assert llm.output_processor.parent_requests == {}


def test_async_add_request_keeps_state_when_engine_accepts():
    llm = _make_async_llm(AsyncMock())
    request = _engine_request()
    queue = RequestOutputCollector(RequestOutputKind.CUMULATIVE, request.request_id)

    asyncio.run(llm._add_request(request, None, None, 0, queue))

    assert request.request_id in llm.output_processor.request_states


def test_async_parallel_add_rolls_back_siblings_on_later_send_failure():
    calls = 0

    async def add_request_async(_request: EngineCoreRequest) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OverflowError("can't serialize ints")

    llm = _make_async_llm(add_request_async)
    params = SamplingParams(n=2, detokenize=False, max_tokens=1)
    request = _engine_request(sampling_params=params)

    async def _run() -> None:
        with pytest.raises(OverflowError):
            await llm.add_request("req-0", request, params)

    asyncio.run(_run())

    assert llm.output_processor.request_states == {}
    assert llm.output_processor.parent_requests == {}


def test_sync_add_request_rolls_back_on_engine_send_failure():
    encoder = MsgpackEncoder()

    def add_request(request: EngineCoreRequest) -> None:
        encoder.encode(request)

    engine = _make_llm_engine(add_request)
    request = _engine_request()
    request.priority = _OVERFLOW

    with pytest.raises(OverflowError):
        engine.add_request("req-0", request, request.sampling_params)

    assert engine.output_processor.request_states == {}
    assert engine.output_processor.parent_requests == {}


def test_sync_parallel_add_rolls_back_siblings_on_later_send_failure():
    calls = 0

    def add_request(_request: EngineCoreRequest) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OverflowError("can't serialize ints")

    engine = _make_llm_engine(add_request)
    params = SamplingParams(n=2, detokenize=False, max_tokens=1)
    request = _engine_request(sampling_params=params)

    with pytest.raises(OverflowError):
        engine.add_request("req-0", request, params)

    assert engine.output_processor.request_states == {}
    assert engine.output_processor.parent_requests == {}
