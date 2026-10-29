# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.aux_output_connector.hidden_states import HiddenStateCapturePlan
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

pytestmark = pytest.mark.cpu_test


def engine_stub():
    engine = MagicMock()
    engine.errored = False
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        aux_output_config=SimpleNamespace(enable_hidden_states=True),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
    )
    return engine


def capture_params(**kwargs):
    plan = HiddenStateCapturePlan.from_window("capture", 2, 0, 1, "test")
    return SamplingParams(extra_args={"hidden_state_capture": asdict(plan)}, **kwargs)


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("case", ["n", "stop", "resumable", "disabled"])
def test_direct_request_validates_effective_capture_params(async_engine, case):
    engine = engine_stub()
    params = capture_params(
        **({"n": 2} if case == "n" else {"stop": "END"} if case == "stop" else {})
    )
    if case == "disabled":
        engine.vllm_config.aux_output_config.enable_hidden_states = False
    request = EngineCoreRequest(
        request_id="capture",
        prompt_token_ids=[1, 2],
        sampling_params=params,
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        arrival_time=0.0,
        resumable=case == "resumable",
    )
    with pytest.raises(VLLMValidationError):
        if async_engine:
            asyncio.run(
                AsyncLLM.add_request(engine, "capture", request, SamplingParams())
            )
        else:
            LLMEngine.add_request(engine, "capture", request, SamplingParams())
    engine.engine_core.add_request.assert_not_called()
    engine.engine_core.add_request_async.assert_not_called()


def test_streaming_capture_rejected_before_consuming_input():
    engine = engine_stub()

    async def stream():
        raise AssertionError("input must not be consumed")
        yield

    inputs = stream()
    try:
        with pytest.raises(VLLMValidationError, match="streaming input"):
            asyncio.run(
                AsyncLLM.add_request(engine, "capture", inputs, capture_params())
            )
        engine._add_streaming_input_request.assert_not_called()
    finally:
        asyncio.run(inputs.aclose())
