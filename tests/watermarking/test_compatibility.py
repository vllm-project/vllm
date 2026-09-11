# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm import SamplingParams
from vllm import logger as vllm_logger
from vllm.config import VllmConfig, WatermarkConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers import BaseRenderer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.llm_engine import LLMEngine


def _input_processor(server_uses_watermarking: bool = True) -> InputProcessor:
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        try_get_generation_config=lambda: {},
        return_sampling_mask=False,
        enable_trace_replay=True,
        is_multimodal_model=False,
    )
    config.watermark_config = (
        WatermarkConfig(key=42) if server_uses_watermarking else None
    )
    renderer = MagicMock(spec=BaseRenderer)
    renderer.tokenizer = None
    renderer._executor = None
    return InputProcessor(config, renderer)


def _validate(params: SamplingParams, server_uses_watermarking: bool = True) -> None:
    processor = _input_processor(server_uses_watermarking)
    with patch.object(SamplingParams, "verify"):
        processor._validate_params(params, ("generate",))


def _engine_core_request(params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="request",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def reset_warning_once():
    vllm_logger._print_warning_once.cache_clear()
    yield
    vllm_logger._print_warning_once.cache_clear()


def test_greedy_watermarked_requests_warn_once(caplog_vllm, reset_warning_once):
    first = SamplingParams(temperature=0)
    second = SamplingParams(temperature=0)

    with caplog_vllm.at_level("WARNING"):
        _validate(first)
        _validate(second)

    message = "subsequent greedy requests will use ordinary greedy sampling"
    assert caplog_vllm.text.count(message) == 1
    assert not first.watermarking
    assert not second.watermarking


def test_inherited_watermarking_resolves_from_engine_configuration():
    enabled = SamplingParams()
    disabled = SamplingParams()

    _validate(enabled)
    _validate(disabled, server_uses_watermarking=False)

    assert enabled.watermarking is True
    assert disabled.watermarking is False


def test_explicit_watermarking_requires_engine_configuration():
    with pytest.raises(VLLMValidationError, match="engine has no watermark"):
        _validate(SamplingParams(watermarking=True), server_uses_watermarking=False)


def test_explicit_watermarking_rejects_greedy_decoding():
    with pytest.raises(VLLMValidationError, match="explicitly requested"):
        _validate(SamplingParams(temperature=0, watermarking=True))


def test_trace_replay_with_watermarking_is_rejected():
    with pytest.raises(VLLMValidationError, match="Trace replay"):
        _validate(SamplingParams(trace_decode_token_ids=[1]))


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0, watermarking=False),
        SamplingParams(trace_decode_token_ids=[1], watermarking=False),
    ],
)
def test_incompatible_modes_are_allowed_when_watermarking_is_disabled(params):
    _validate(params)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0),
        SamplingParams(trace_decode_token_ids=[1]),
    ],
)
def test_watermarking_checks_are_inactive_without_engine_config(params):
    _validate(params, server_uses_watermarking=False)
    assert not params.watermarking


def test_trace_replay_still_rejects_greedy_requests():
    with pytest.raises(VLLMValidationError, match="Trace replay"):
        _validate(SamplingParams(temperature=0, trace_decode_token_ids=[1]))


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(seed=42),
        SamplingParams(n=2),
        SamplingParams(n=2, seed=42),
    ],
)
def test_seeded_and_parallel_sampling_are_allowed(params):
    _validate(params)


def test_direct_engine_request_is_rejected_before_id_mutation():
    engine = object.__new__(LLMEngine)
    engine.input_processor = _input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    with pytest.raises(VLLMValidationError, match="Trace replay"):
        engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    assert request.request_id == "request"
    assert request.external_req_id is None


def test_direct_engine_request_uses_embedded_opt_out(monkeypatch):
    engine = object.__new__(LLMEngine)
    engine.input_processor = _input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    assign_request_id = MagicMock()
    monkeypatch.setattr(engine.input_processor, "assign_request_id", assign_request_id)
    engine.output_processor = MagicMock()
    engine.engine_core = MagicMock()
    request = _engine_core_request(SamplingParams(temperature=0, watermarking=False))

    engine.add_request(
        request.request_id,
        request,
        SamplingParams(temperature=0),
    )

    assign_request_id.assert_called_once_with(request)


def test_direct_async_engine_request_is_rejected_before_side_effects():
    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False), shutdown=lambda **kwargs: None
    )
    engine.output_handler = None
    engine.input_processor = _input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    async def add_request() -> None:
        await engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    with pytest.raises(VLLMValidationError, match="Trace replay"):
        asyncio.run(add_request())

    assert request.request_id == "request"
    assert request.external_req_id is None
    assert engine.output_handler is None


def test_direct_async_engine_greedy_request_warns_and_disables_watermarking(
    caplog_vllm, monkeypatch, reset_warning_once
):
    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        shutdown=lambda **kwargs: None,
        add_request_async=AsyncMock(),
    )
    engine.output_handler = None
    engine.input_processor = _input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    engine.output_processor = MagicMock()
    engine.log_requests = False
    monkeypatch.setattr(engine, "_run_output_handler", MagicMock())
    request = _engine_core_request(SamplingParams(temperature=0))

    async def add_request() -> None:
        await engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    with caplog_vllm.at_level("WARNING"):
        asyncio.run(add_request())

    assert "subsequent greedy requests will use ordinary greedy sampling" in (
        caplog_vllm.text
    )
    assert not request.sampling_params.watermarking
