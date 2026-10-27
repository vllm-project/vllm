# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from copy import copy
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
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest


def _input_processor(server_uses_watermarking: bool = True) -> InputProcessor:
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        try_get_generation_config=lambda: {},
        logits_processors=None,
        return_sampling_mask=False,
        enable_trace_replay=True,
        is_multimodal_model=False,
        supports_multimodal_inputs=False,
        runner_type="generate",
        is_diffusion=False,
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


@pytest.mark.parametrize("watermarking", [True])
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
        SamplingParams(),
        SamplingParams(temperature=0),
        SamplingParams(trace_decode_token_ids=[1]),
    ],
)
def test_requests_without_engine_config_are_unwatermarked(
    params, caplog_vllm, reset_warning_once
):
    with caplog_vllm.at_level("WARNING"):
        resolved = _validate(params, server_uses_watermarking=False)

    assert "engine has no watermark configuration" not in caplog_vllm.text
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
    assert request.sampling_params._watermarking_skipped is True
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

    assert request.sampling_params._watermarking_skipped is False
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
    assert request.sampling_params._watermarking_skipped is True


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
    assert request.sampling_params._watermarking_skipped is True


@pytest.mark.parametrize(
    ("kwargs", "server_uses_watermarking", "skipped"),
    [
        ({}, True, False),
        ({"temperature": 0}, True, True),
        ({"temperature": 0, "watermarking": False}, True, False),
        ({"temperature": 0}, False, False),
    ],
)
def test_apply_watermarking_records_skipped_requests(
    kwargs, server_uses_watermarking, skipped, reset_warning_once
):
    params = SamplingParams(**kwargs)
    processor = _input_processor(server_uses_watermarking)
    processor.apply_watermarking(params, processor.resolve_watermarking(params))
    assert params._watermarking_skipped is skipped

    # Resolving an already resolved request keeps the flag.
    processor.apply_watermarking(params, processor.resolve_watermarking(params))
    assert params._watermarking_skipped is skipped


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


def _enforced_input_processor(enforce: bool = True) -> InputProcessor:
    processor = _input_processor()
    processor.vllm_config.watermark_config = WatermarkConfig(key=42, enforce=enforce)
    return processor


@pytest.mark.parametrize(
    ("kwargs", "enforce", "resolved", "skipped"),
    [
        pytest.param({"watermarking": False}, True, True, False, id="opt-out"),
        pytest.param({}, True, True, False, id="default"),
        pytest.param({"watermarking": True}, True, True, False, id="opt-in"),
        # Internal beam-search step and generative-scoring requests opt out
        # with temperature > 0; enforce watermarks them like any opt-out.
        pytest.param(
            {"watermarking": False, "max_tokens": 1, "logprobs": 4},
            True,
            True,
            False,
            id="internal-opt-out",
        ),
        pytest.param(
            {"temperature": 0, "watermarking": False},
            True,
            False,
            False,
            id="greedy-opt-out",
        ),
        pytest.param({"temperature": 0}, True, False, True, id="greedy-default"),
        # top_k=1 resolves to True although its output equals greedy output.
        pytest.param(
            {"top_k": 1, "watermarking": False}, True, True, False, id="top-k-1"
        ),
        pytest.param(
            {"trace_decode_token_ids": [1], "watermarking": False},
            True,
            False,
            False,
            id="trace-replay-opt-out",
        ),
        pytest.param(
            {"trace_decode_token_ids": [1]},
            True,
            False,
            True,
            id="trace-replay-default",
        ),
        pytest.param(
            {"watermarking": False}, False, False, False, id="opt-out-enforce-off"
        ),
        pytest.param({}, False, True, False, id="default-enforce-off"),
        pytest.param(
            {"temperature": 0, "watermarking": False},
            False,
            False,
            False,
            id="greedy-opt-out-enforce-off",
        ),
    ],
)
def test_enforce_resolves_opt_outs_in_frontend(
    kwargs, enforce, resolved, skipped, reset_warning_once
):
    processor = _enforced_input_processor(enforce)
    params = SamplingParams(**kwargs)

    with patch.object(SamplingParams, "verify"):
        assert processor._validate_params(params, ("generate",)) is resolved
    # The caller's params object is left untouched.
    assert params.watermarking is kwargs.get("watermarking")

    processor.apply_watermarking(params, processor.resolve_watermarking(params))
    assert params.watermarking is resolved
    assert params._watermarking_skipped is skipped

    # Resolving an already resolved request keeps both values.
    processor.apply_watermarking(params, processor.resolve_watermarking(params))
    assert params.watermarking is resolved
    assert params._watermarking_skipped is skipped


@pytest.mark.parametrize("watermarking", [None, False])
def test_enforce_keeps_beam_search_unwatermarked(
    watermarking, caplog_vllm, reset_warning_once
):
    params = BeamSearchParams(
        beam_width=2, max_tokens=4, temperature=0.8, watermarking=watermarking
    )

    with caplog_vllm.at_level("WARNING"):
        assert _enforced_input_processor().resolve_watermarking(params) is False
    assert "beam search requests will run without watermarking" in caplog_vllm.text
    assert params.watermarking is watermarking


def test_enforce_greedy_opt_out_warns(caplog_vllm, reset_warning_once):
    processor = _enforced_input_processor()
    params = SamplingParams(temperature=0, watermarking=False)

    with caplog_vllm.at_level("WARNING"):
        assert processor.resolve_watermarking(params) is False
    assert "ordinary greedy sampling" in caplog_vllm.text


def test_direct_engine_request_opt_out_is_enforced(monkeypatch):
    engine = object.__new__(LLMEngine)
    engine.input_processor = _enforced_input_processor()
    engine.vllm_config = engine.input_processor.vllm_config
    monkeypatch.setattr(engine.input_processor, "assign_request_id", MagicMock())
    engine.output_processor = MagicMock()
    engine.engine_core = MagicMock()
    request = _engine_core_request(SamplingParams(watermarking=False))

    engine.add_request(request.request_id, request, SamplingParams())

    # The output processor sees the enforced value, so metrics count it.
    assert request.sampling_params.watermarking is True
    assert request.sampling_params._watermarking_skipped is False
    engine.output_processor.add_request.assert_called_once()


def _preprocess_in_engine_core(
    params: SamplingParams, watermark_config: WatermarkConfig | None
) -> SamplingParams:
    from vllm.v1.engine.core import EngineCore

    core = object.__new__(EngineCore)
    core.vllm_config = VllmConfig()
    core.vllm_config.watermark_config = watermark_config
    core.mm_receiver_cache = None
    core.request_block_hasher = None
    request = _engine_core_request(params)
    core.preprocess_add_request(request)
    return request.sampling_params


# EngineCore applies enforce again for frontends that never ran the check
# (Rust HTTP and gRPC), so these params arrive unresolved.
@pytest.mark.parametrize(
    ("kwargs", "watermark_config", "expected"),
    [
        pytest.param(
            {"watermarking": False},
            WatermarkConfig(key=42, enforce=True),
            True,
            id="opt-out",
        ),
        pytest.param(
            {"watermarking": True},
            WatermarkConfig(key=42, enforce=True),
            True,
            id="opt-in",
        ),
        pytest.param(
            {"temperature": 0, "watermarking": False},
            WatermarkConfig(key=42, enforce=True),
            True,
            id="greedy",
        ),
        pytest.param(
            {"trace_decode_token_ids": [1], "watermarking": False},
            WatermarkConfig(key=42, enforce=True),
            True,
            id="trace-replay",
        ),
        pytest.param(
            {"watermarking": False},
            WatermarkConfig(key=42),
            False,
            id="enforce-off",
        ),
        pytest.param({"watermarking": False}, None, False, id="no-config"),
    ],
)
def test_enforce_engine_core_backstop(kwargs, watermark_config, expected):
    params = _preprocess_in_engine_core(SamplingParams(**kwargs), watermark_config)
    assert params.watermarking is expected


def test_enforce_override_warns_once_across_admission_paths(
    caplog_vllm, reset_warning_once
):
    config = VllmConfig()
    config.watermark_config = WatermarkConfig(key=987654321, enforce=True)
    params = SamplingParams(watermarking=False)

    with caplog_vllm.at_level("WARNING"):
        params.watermarking = config._check_supports_watermarking(params)
        _preprocess_in_engine_core(params, config.watermark_config)
        _preprocess_in_engine_core(
            SamplingParams(watermarking=False), config.watermark_config
        )
        config._check_supports_watermarking(SamplingParams(watermarking=False))

    assert params.watermarking is True
    assert caplog_vllm.text.count("overriding watermarking=False") == 1
    assert "987654321" not in caplog_vllm.text


def test_enforce_core_override_warns_once(caplog_vllm, reset_warning_once):
    config = WatermarkConfig(key=987654321, enforce=True)
    with caplog_vllm.at_level("WARNING"):
        for _ in range(2):
            params = _preprocess_in_engine_core(
                SamplingParams(watermarking=False), config
            )
            assert params.watermarking is True

    assert caplog_vllm.text.count("overriding watermarking=False") == 1
    assert "987654321" not in caplog_vllm.text


@pytest.mark.parametrize("watermarking", [None, True, False])
@pytest.mark.parametrize("enforce", [False, True])
def test_enforce_warning_only_for_overridden_false(
    watermarking, enforce, caplog_vllm, reset_warning_once
):
    config = VllmConfig()
    config.watermark_config = WatermarkConfig(key=42, enforce=enforce)
    with caplog_vllm.at_level("WARNING"):
        config._check_supports_watermarking(SamplingParams(watermarking=watermarking))
        _preprocess_in_engine_core(
            SamplingParams(watermarking=watermarking), config.watermark_config
        )

    assert ("overriding watermarking=False" in caplog_vllm.text) is (
        enforce and watermarking is False
    )


@pytest.mark.parametrize("enforce", [False, True])
def test_parallel_opt_out_stats_use_resolved_watermarking(enforce):
    config = VllmConfig()
    config.watermark_config = WatermarkConfig(key=42, enforce=enforce)
    params = SamplingParams(n=4, watermarking=False, detokenize=False)
    params.watermarking = config._check_supports_watermarking(params)
    request = _engine_core_request(params)
    request.external_req_id = request.request_id
    parent = ParentRequest(request)
    output_processor = OutputProcessor(None, log_stats=True)

    for index in range(params.n):
        child = copy(request)
        child.request_id, child.sampling_params = parent.get_child_info(index)
        output_processor.add_request(child, None, parent, index)
        _preprocess_in_engine_core(child.sampling_params, config.watermark_config)

    stats = [state.stats for state in output_processor.request_states.values()]
    assert len(stats) == 4
    assert all(stat is not None and stat.is_watermarked is enforce for stat in stats)
