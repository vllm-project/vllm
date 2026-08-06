# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for admission control (max_num_queued_reqs / max_num_queued_tokens).

These tests cover:
- OutputProcessor.get_num_queued_tokens() token counting
- AsyncLLM.check_admission() admission control logic
- Exception classes (GracefulHTTPError, QueueOverflowError, MaxQueuedTokensError)
- create_error_response() mapping GracefulHTTPError to HTTP 503
- SchedulerConfig field defaults and validation
- human_readable_int CLI notation for max_num_queued_tokens
"""

import argparse
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from vllm.config.scheduler import SchedulerConfig
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
    VLLMError,
)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import human_readable_int
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_req_state(prompt_len: int, is_prefilling: bool = True):
    return SimpleNamespace(
        prompt_len=prompt_len,
        is_prefilling=is_prefilling,
    )


def _make_async_llm(
    max_num_queued_reqs: int | None = None,
    max_num_queued_tokens: int | None = None,
    num_unfinished: int = 0,
    num_queued_tokens: int = 0,
) -> AsyncLLM:
    """Create a bare AsyncLLM with just the attributes needed for scheduling."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=max_num_queued_reqs,
        max_num_queued_tokens=max_num_queued_tokens,
    )
    llm.output_processor = MagicMock()
    llm.output_processor.get_num_unfinished_requests.return_value = num_unfinished
    llm.output_processor.get_num_queued_tokens.return_value = num_queued_tokens
    return llm


def _make_output_processor(**request_states) -> OutputProcessor:
    op = OutputProcessor.__new__(OutputProcessor)
    op.request_states = request_states
    return op


def _make_scheduler_config(**kwargs) -> SchedulerConfig:
    return SchedulerConfig(
        runner_type="generate",
        max_model_len=4096,
        is_encoder_decoder=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_graceful_http_error_carries_status_and_message():
    err = GracefulHTTPError("custom message", HTTPStatus.SERVICE_UNAVAILABLE)
    assert err.message == "custom message"
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert str(err) == "custom message"


def test_graceful_http_error_is_vllm_error():
    err = GracefulHTTPError("msg", HTTPStatus.TOO_MANY_REQUESTS)
    assert isinstance(err, VLLMError)


def test_queue_overflow_error():
    err = QueueOverflowError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "busy" in err.message.lower() or "try again" in err.message.lower()


def test_max_queued_tokens_error():
    err = MaxQueuedTokensError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "backlog" in err.message.lower() or "try again" in err.message.lower()


@pytest.mark.parametrize("exc_cls", [QueueOverflowError, MaxQueuedTokensError])
def test_admission_exceptions_are_vllm_errors(exc_cls):
    """Admission rejections must reach the VLLMError HTTP handler."""
    assert issubclass(exc_cls, VLLMError)


# ---------------------------------------------------------------------------
# OutputProcessor.get_num_queued_tokens
# ---------------------------------------------------------------------------


def test_queued_tokens_empty():
    assert _make_output_processor().get_num_queued_tokens() == 0


def test_queued_tokens_sums_prefilling_requests():
    op = _make_output_processor(r1=_make_req_state(100), r2=_make_req_state(200))
    assert op.get_num_queued_tokens() == 300


def test_queued_tokens_excludes_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=True),
        r2=_make_req_state(200, is_prefilling=False),
        r3=_make_req_state(50, is_prefilling=True),
    )
    assert op.get_num_queued_tokens() == 150


def test_queued_tokens_all_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=False),
        r2=_make_req_state(200, is_prefilling=False),
    )
    assert op.get_num_queued_tokens() == 0


# ---------------------------------------------------------------------------
# AsyncLLM.check_admission
# ---------------------------------------------------------------------------


def test_admission_no_limits_allows_everything():
    llm = _make_async_llm(num_unfinished=999, num_queued_tokens=999)
    llm.check_admission()


# -- max_num_queued_reqs ----------------------------------------------------


def test_admission_reqs_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=5)
    llm.check_admission()


def test_admission_reqs_rejects_at_limit():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


def test_admission_reqs_rejects_with_n():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=8)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(3)


def test_admission_reqs_allows_n_at_boundary():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=7)
    llm.check_admission(3)


def test_admission_reqs_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_reqs=0, num_unfinished=0)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


# -- max_num_queued_tokens --------------------------------------------------


def test_admission_tokens_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=500)
    llm.check_admission()


def test_admission_tokens_rejects_at_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1000)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_tokens_rejects_over_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1500)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_tokens_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_tokens=0, num_queued_tokens=0)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


# -- interaction between both limits ----------------------------------------


def test_admission_both_limits_checked_independently():
    llm = _make_async_llm(
        max_num_queued_reqs=100,
        max_num_queued_tokens=1000,
        num_unfinished=5,
        num_queued_tokens=1000,
    )
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_req_limit_checked_before_token_limit():
    llm = _make_async_llm(
        max_num_queued_reqs=10,
        max_num_queued_tokens=1000,
        num_unfinished=10,
        num_queued_tokens=1000,
    )
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


# -- n derived from params at the add_request call site ---------------------


@pytest.mark.parametrize("params", [SamplingParams(), PoolingParams()])
def test_admission_params_without_explicit_n_count_as_one_slot(params):
    """PoolingParams has no ``.n``; add_request must fall back to 1."""
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=9)
    llm.check_admission(getattr(params, "n", 1) or 1)

    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(getattr(params, "n", 1) or 1)


# ---------------------------------------------------------------------------
# create_error_response integration
# ---------------------------------------------------------------------------


def test_queue_overflow_maps_to_503():
    resp = create_error_response(QueueOverflowError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    assert resp.error.param is None
    msg = resp.error.message.lower()
    assert "busy" in msg or "try again" in msg


def test_max_queued_tokens_maps_to_503():
    resp = create_error_response(MaxQueuedTokensError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    msg = resp.error.message.lower()
    assert "backlog" in msg or "try again" in msg


def test_custom_graceful_error_maps_to_its_status():
    err = GracefulHTTPError("custom", HTTPStatus.SERVICE_UNAVAILABLE)
    resp = create_error_response(err)
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase


# ---------------------------------------------------------------------------
# SchedulerConfig field defaults
# ---------------------------------------------------------------------------


def test_scheduler_config_defaults_are_none():
    config = _make_scheduler_config()
    assert config.max_num_queued_reqs is None
    assert config.max_num_queued_tokens is None


def test_scheduler_config_accepts_explicit_values():
    config = _make_scheduler_config(
        max_num_queued_reqs=100,
        max_num_queued_tokens=32000,
    )
    assert config.max_num_queued_reqs == 100
    assert config.max_num_queued_tokens == 32000


@pytest.mark.parametrize("field", ["max_num_queued_reqs", "max_num_queued_tokens"])
def test_scheduler_config_rejects_negative(field):
    with pytest.raises(ValidationError):
        _make_scheduler_config(**{field: -1})


# ---------------------------------------------------------------------------
# human_readable_int for CLI notation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("32k", 32_000),
        ("1k", 1_000),
        ("1K", 1_024),
        ("1m", 1_000_000),
        ("1M", 1_048_576),
        ("100", 100),
        ("2.5k", 2_500),
        ("0", 0),
    ],
)
def test_human_readable_int_parses_notation(input_str: str, expected: int):
    assert human_readable_int(input_str) == expected


@pytest.mark.parametrize("invalid", ["abc", "1x", "", "k", "1.5K"])
def test_human_readable_int_rejects_invalid(invalid: str):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        human_readable_int(invalid)
