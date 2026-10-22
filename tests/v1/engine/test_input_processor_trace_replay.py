# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.skip_global_cleanup


def _make_request(params: SamplingParams) -> Request:
    return Request(
        request_id="req-0",
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
    )


def _normalize(params: SamplingParams, prompt_len: int, max_model_len: int) -> None:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len)
    )
    InputProcessor._normalize_trace_replay_params(processor, params, prompt_len)


def test_normalize_trace_replay_params():
    params = SamplingParams(
        max_tokens=16,
        min_tokens=10,
        stop_token_ids=[20],
        trace_decode_token_ids=[10, 20, 30],
    )
    params.update_from_generation_config({}, eos_token_id=20)

    _normalize(params, prompt_len=3, max_model_len=128)

    assert params.max_tokens == 3
    assert params.min_tokens == 0
    assert params.ignore_eos is True
    assert params.eos_token_id is None
    assert params.stop_token_ids == []
    assert params.all_stop_token_ids == set()

    request = _make_request(params)
    assert request.max_tokens == 3
    request.append_output_token_ids([10, 20, 30])
    assert check_stop(request, max_model_len=128)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_trace_longer_than_remaining_context_is_truncated():
    """The trace is staged into a max_model_len-wide row, so it must be cut.

    An explicitly set max_tokens is never clamped to max_model_len, so it cannot
    bound the staged write.
    """
    params = SamplingParams(max_tokens=100, trace_decode_token_ids=list(range(20)))

    _normalize(params, prompt_len=6, max_model_len=10)

    assert params.trace_decode_token_ids == [0, 1, 2, 3]
    assert params.max_tokens == 4


def _validate(
    enable_trace_replay: bool,
    params: SamplingParams | None = None,
    kda_recoverssm_prefill_only: bool = False,
) -> None:
    """Run _validate_params' trace gating with the rest of verify() stubbed."""
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            return_sampling_mask=False,
            enable_trace_replay=enable_trace_replay,
            is_diffusion=False,
        ),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                kda_recoverssm_prefill_only=kda_recoverssm_prefill_only
            ),
            reasoning_config=None,
            # Engine has no watermark configuration.
            _check_supports_watermarking=lambda params: False,
        ),
        resolve_watermarking=lambda params: False,
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=None,
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
    )
    if params is None:
        params = SamplingParams(trace_decode_token_ids=[1, 2, 3])
    with patch.object(SamplingParams, "verify"):
        InputProcessor._validate_params(processor, params, ("generate",))


def test_trace_request_rejected_when_feature_disabled():
    with pytest.raises(VLLMValidationError, match="--enable-trace-replay"):
        _validate(enable_trace_replay=False)


def test_trace_request_accepted_when_feature_enabled():
    _validate(enable_trace_replay=True)


def test_trace_replay_requires_v2_model_runner():
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_trace_replay=True),
        use_v2_model_runner=False,
    )

    with pytest.raises(ValueError, match="trace replay requires Model Runner V2"):
        VllmConfig._verify_trace_replay_config(config)


@pytest.mark.parametrize("max_tokens", [None, 2, 16])
def test_prefill_only_instance_rejects_decode_requests(max_tokens):
    """A --kda-recoverssm-prefill-only instance has no room for verify records,
    so requests that would decode (and verify draft tokens) there are rejected
    at admission."""
    params = SamplingParams(max_tokens=max_tokens)
    with pytest.raises(VLLMValidationError, match="max_tokens=1"):
        _validate(False, params, kda_recoverssm_prefill_only=True)
    _validate(False, params, kda_recoverssm_prefill_only=False)


def test_prefill_only_instance_accepts_prefill_requests():
    _validate(False, SamplingParams(max_tokens=1), kda_recoverssm_prefill_only=True)
