# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for streaming-input session lifecycle hardening:
post-abort chunk races, unbounded resume, zero-chunk streams, and DP
routing."""

from __future__ import annotations

import asyncio
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.engine.protocol import StreamingInput
from vllm.outputs import STREAM_FINISHED
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

STOP_TOKEN = 128001


def _create_scheduler() -> Scheduler:
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.model_config.is_multimodal_model = False
    vllm_config.model_config.is_encoder_decoder = False
    vllm_config.model_config.max_model_len = 1024
    vllm_config.model_config.enable_return_routed_experts = False
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.num_gpu_blocks = 1000
    vllm_config.cache_config.enable_prefix_caching = False
    kv_cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
        hash_block_size=16,
    )


def _make_chunk(
    request_id: str,
    tokens: list[int],
    max_tokens: int = 16,
    first_chunk: bool = True,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(
            stop_token_ids=[STOP_TOKEN], max_tokens=max_tokens
        ),
        pooling_params=None,
        resumable=True,
        first_chunk=first_chunk,
    )


def _mro(req_id: str, token: int) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token]],
        logprobs=None,
        prompt_logprobs_dict={req_id: None},
        pooler_output=[],
    )


def test_non_first_chunk_for_unknown_session_is_dropped():
    """Only a session's first chunk may create a session: a non-first chunk
    for an id the scheduler does not know belongs to a finished (or foreign)
    session, and admitting it would resurrect an unabortable request."""
    scheduler = _create_scheduler()

    # Cold scheduler: a mid-session chunk with no known session is dropped.
    scheduler.add_request(_make_chunk("sess-x", [7, 8], first_chunk=False))
    assert "sess-x" not in scheduler.requests
    assert len(scheduler.waiting) == 0
    assert scheduler.num_waiting_for_streaming_input == 0

    # The race this guards: session finishes, its late chunk must not revive.
    scheduler.add_request(_make_chunk("sess-t", [1, 2, 3]))
    scheduler.finish_requests("sess-t", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_make_chunk("sess-t", [7, 8], first_chunk=False))
    assert "sess-t" not in scheduler.requests

    # First chunks still open sessions.
    scheduler.add_request(_make_chunk("sess-b", [4, 5, 6]))
    assert "sess-b" in scheduler.requests


@pytest.mark.asyncio
async def test_stop_input_stream_cancels_and_awaits():
    """No chunk send may complete after _stop_input_stream returns."""
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id="r1")
    sent = []

    async def slow_send():
        await asyncio.sleep(30)
        sent.append("chunk")

    task = asyncio.create_task(slow_send())
    queue._input_stream_task = task
    await asyncio.sleep(0)

    await AsyncLLM._stop_input_stream(queue)

    assert task.cancelled()
    assert queue._input_stream_task is None
    assert sent == []


@pytest.mark.asyncio
async def test_handle_inputs_stamps_first_chunk():
    """The frontend marks exactly the first chunk of a session."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._validate_streaming_input_sampling_params = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    llm._run_output_handler = MagicMock()
    llm._add_request = AsyncMock()
    llm.model_config = MagicMock()

    reqs = [MagicMock(prompt_embeds=None) for _ in range(3)]
    reqs[0].request_id = "r1-int"
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.side_effect = reqs
    llm.input_processor.assign_request_id = MagicMock()

    async def two_chunks():
        yield StreamingInput(prompt="frame-1")
        yield StreamingInput(prompt="frame-2")

    with mock.patch(
        "vllm.v1.engine.async_llm.extract_prompt_components",
        return_value=("", None, None),
    ):
        await llm._add_streaming_input_request(
            "r1", two_chunks(), SamplingParams(max_tokens=4)
        )
        for _ in range(6):
            await asyncio.sleep(0)

    assert reqs[1].first_chunk is True
    assert reqs[2].first_chunk is False


@pytest.mark.asyncio
async def test_input_stream_task_cleared_only_after_final_send():
    """The task pointer must stay set until the final request is sent, so a
    concurrent abort can still cancel-and-await the task (no send may ever
    happen after an abort)."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._validate_streaming_input_sampling_params = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    llm._run_output_handler = MagicMock()
    llm.model_config = MagicMock()

    reqs = [MagicMock(prompt_embeds=None) for _ in range(2)]
    reqs[0].request_id = "r1-int"
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.side_effect = reqs
    llm.input_processor.assign_request_id = MagicMock()

    holder = {}
    pointer_set_during_send = []

    async def capture_add_request(*args, **kwargs):
        pointer_set_during_send.append(holder["queue"]._input_stream_task is not None)

    llm._add_request = AsyncMock(side_effect=capture_add_request)

    async def one_chunk():
        yield StreamingInput(prompt="frame-1")

    with mock.patch(
        "vllm.v1.engine.async_llm.extract_prompt_components",
        return_value=("", None, None),
    ):
        holder["queue"] = await llm._add_streaming_input_request(
            "r1", one_chunk(), SamplingParams(max_tokens=4)
        )
        for _ in range(6):
            await asyncio.sleep(0)

    # Two sends (the chunk, then the final request); the pointer must have
    # been set for BOTH, and cleared only once everything was sent.
    assert pointer_set_during_send == [True, True]
    assert holder["queue"]._input_stream_task is None


@pytest.mark.asyncio
async def test_zero_chunk_stream_emits_finished_sentinel():
    """An input stream that closes without chunks must unblock the consumer
    instead of submitting the placeholder final request as a generation."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._validate_streaming_input_sampling_params = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    llm._run_output_handler = MagicMock()
    llm._add_request = AsyncMock()
    llm.model_config = MagicMock()

    final_req = MagicMock()
    final_req.request_id = "r1-int"
    final_req.prompt_embeds = None
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.return_value = final_req
    llm.input_processor.assign_request_id = MagicMock()

    async def empty_stream():
        return
        yield  # pragma: no cover

    queue = await llm._add_streaming_input_request(
        "r1", empty_stream(), SamplingParams(max_tokens=4)
    )
    for _ in range(4):
        await asyncio.sleep(0)

    assert queue.get_nowait() is STREAM_FINISHED
    llm._add_request.assert_not_awaited()


def test_dp_sticky_routing_for_in_flight_request():
    """Chunks of an in-flight session route to the engine holding it."""
    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    engine = object()
    client.reqs_in_flight = {"sess-1": engine}

    request = MagicMock()
    request.request_id = "sess-1"
    assert client.get_core_engine_for_request(request) is engine
