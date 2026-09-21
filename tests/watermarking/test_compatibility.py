# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm import SamplingParams
from vllm import logger as vllm_logger
from vllm.config import VllmConfig, WatermarkConfig
from vllm.renderers import BaseRenderer
from vllm.sampling_params import BeamSearchParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.llm_engine import LLMEngine


def _input_processor(server_uses_watermarking: bool = True) -> InputProcessor:
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        try_get_generation_config=lambda: {},
        logits_processors=None,
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


def _validate(
    params: SamplingParams, server_uses_watermarking: bool = True
) -> bool | None:
    processor = _input_processor(server_uses_watermarking)
    with patch.object(SamplingParams, "verify"):
        return processor._validate_params(params, ("generate",))


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
        first_resolved = _validate(first)
        second_resolved = _validate(second)

    message = "subsequent greedy requests will use ordinary greedy sampling"
    assert caplog_vllm.text.count(message) == 1
    assert first_resolved is False
    assert second_resolved is False
    # The caller's params objects are left untouched.
    assert first.watermarking is None
    assert second.watermarking is None


def test_default_watermarking_resolves_when_configured():
    enabled = SamplingParams()

    assert _validate(enabled) is True
    # The caller's params object keeps inheriting.
    assert enabled.watermarking is None


@pytest.mark.parametrize("watermarking", [None, True])
def test_requested_watermarking_without_engine_config_warns_and_disables(
    watermarking, caplog_vllm, reset_warning_once
):
    params = SamplingParams(watermarking=watermarking)

    with caplog_vllm.at_level("WARNING"):
        resolved = _validate(params, server_uses_watermarking=False)

    assert "engine has no watermark configuration" in caplog_vllm.text
    assert resolved is False
    assert params.watermarking is watermarking


@pytest.mark.parametrize("watermarking", [None, True])
def test_explicit_and_default_greedy_warn_and_disable(
    watermarking, caplog_vllm, reset_warning_once
):
    params = SamplingParams(temperature=0, watermarking=watermarking)

    with caplog_vllm.at_level("WARNING"):
        resolved = _validate(params)

    assert "ordinary greedy sampling" in caplog_vllm.text
    assert resolved is False
    assert params.watermarking is watermarking


def test_trace_replay_with_watermarking_warns_and_disables(
    caplog_vllm, reset_warning_once
):
    params = SamplingParams(trace_decode_token_ids=[1])

    with caplog_vllm.at_level("WARNING"):
        resolved = _validate(params)

    assert "trace replay requests will run without watermarking" in caplog_vllm.text
    assert resolved is False
    assert params.watermarking is None


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
def test_requests_without_engine_config_are_unwatermarked(
    params, caplog_vllm, reset_warning_once
):
    with caplog_vllm.at_level("WARNING"):
        resolved = _validate(params, server_uses_watermarking=False)

    assert "engine has no watermark configuration" in caplog_vllm.text
    assert resolved is False
    assert params.watermarking is None


def test_trace_replay_takes_precedence_over_greedy_warning(
    caplog_vllm, reset_warning_once
):
    params = SamplingParams(temperature=0, trace_decode_token_ids=[1])

    with caplog_vllm.at_level("WARNING"):
        _validate(params)

    assert "trace replay requests will run without watermarking" in caplog_vllm.text
    assert "ordinary greedy sampling" not in caplog_vllm.text


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


def test_direct_engine_trace_replay_warns_and_disables(
    caplog_vllm, monkeypatch, reset_warning_once
):
    engine = object.__new__(LLMEngine)
    engine.input_processor = _input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    assign_request_id = MagicMock()
    monkeypatch.setattr(engine.input_processor, "assign_request_id", assign_request_id)
    engine.output_processor = MagicMock()
    engine.engine_core = MagicMock()
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    with caplog_vllm.at_level("WARNING"):
        engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    assert "trace replay requests will run without watermarking" in caplog_vllm.text
    assert request.sampling_params.watermarking is False
    assign_request_id.assert_called_once_with(request)


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


def test_direct_async_engine_trace_replay_warns_and_disables(
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
    request = _engine_core_request(SamplingParams(trace_decode_token_ids=[1]))

    async def add_request() -> None:
        await engine.add_request(
            request.request_id,
            request,
            SamplingParams(watermarking=False),
        )

    with caplog_vllm.at_level("WARNING"):
        asyncio.run(add_request())

    assert "trace replay requests will run without watermarking" in caplog_vllm.text
    assert request.sampling_params.watermarking is False


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
    assert request.sampling_params.watermarking is False


def test_shared_sampling_params_are_not_rewritten_by_admission(reset_warning_once):
    """A params object reused across prompts/engines must keep inheriting."""
    params = SamplingParams(temperature=0.8)

    assert _validate(params, server_uses_watermarking=False) is False
    assert params.watermarking is None
    # Same object submitted to a watermarked engine must still be watermarked.
    assert _validate(params, server_uses_watermarking=True) is True

    greedy = SamplingParams(temperature=0.0)
    assert _validate(greedy) is False
    greedy.temperature = 0.8
    assert _validate(greedy) is True


def test_shared_beam_search_params_are_not_rewritten_by_admission(
    caplog_vllm, reset_warning_once
):
    """Beam search must keep resolving per engine after reuse."""
    params = BeamSearchParams(beam_width=2, max_tokens=4, temperature=0.8)

    # A plain engine leaves the tri-state unresolved on the caller's object.
    assert _input_processor(False).resolve_watermarking(params) is False
    assert params.watermarking is None

    # The same object on a watermarked engine is warned about and still
    # resolves to False, without the first decision being written back.
    with caplog_vllm.at_level("WARNING"):
        assert _input_processor(True).resolve_watermarking(params) is False
    assert "beam search requests will run without watermarking" in caplog_vllm.text
    assert params.watermarking is None
