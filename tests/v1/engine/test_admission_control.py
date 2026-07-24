# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for admission control (max_num_queued_reqs / max_num_queued_tokens).

These tests cover:
- OutputProcessor.get_num_queued_tokens() token counting
- AsyncLLM._validate_request_scheduling() admission control logic
- Exception classes (GracefulHTTPError, QueueOverflowError, MaxQueuedTokensError)
- create_error_response() mapping GracefulHTTPError to HTTP 429
- SchedulerConfig field defaults and validation
- human_readable_int CLI notation for max_num_queued_tokens
"""

import argparse
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import human_readable_int
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor

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


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for GracefulHTTPError and its subclasses."""

    def test_graceful_http_error_carries_status_and_message(self):
        err = GracefulHTTPError("custom message", HTTPStatus.SERVICE_UNAVAILABLE)
        assert err.message == "custom message"
        assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        assert str(err) == "custom message"

    def test_graceful_http_error_is_value_error(self):
        err = GracefulHTTPError("msg", HTTPStatus.TOO_MANY_REQUESTS)
        assert isinstance(err, ValueError)

    def test_queue_overflow_error(self):
        err = QueueOverflowError()
        assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        assert isinstance(err, GracefulHTTPError)
        assert "busy" in err.message.lower() or "try again" in err.message.lower()

    def test_max_queued_tokens_error(self):
        err = MaxQueuedTokensError()
        assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        assert isinstance(err, GracefulHTTPError)
        assert "backlog" in err.message.lower() or "try again" in err.message.lower()

    def test_subclasses_are_value_errors(self):
        """All admission control exceptions are catchable as ValueError."""
        for exc_cls in (QueueOverflowError, MaxQueuedTokensError):
            assert issubclass(exc_cls, ValueError)


# ---------------------------------------------------------------------------
# OutputProcessor.get_num_queued_tokens
# ---------------------------------------------------------------------------


class TestGetNumQueuedTokens:
    """Tests for OutputProcessor.get_num_queued_tokens."""

    def test_empty(self):
        op = OutputProcessor.__new__(OutputProcessor)
        op.request_states = {}
        assert op.get_num_queued_tokens() == 0

    def test_only_prefilling_requests(self):
        op = OutputProcessor.__new__(OutputProcessor)
        op.request_states = {
            "r1": _make_req_state(100),
            "r2": _make_req_state(200),
        }
        assert op.get_num_queued_tokens() == 300

    def test_excludes_non_prefilling(self):
        op = OutputProcessor.__new__(OutputProcessor)
        op.request_states = {
            "r1": _make_req_state(100, is_prefilling=True),
            "r2": _make_req_state(200, is_prefilling=False),
            "r3": _make_req_state(50, is_prefilling=True),
        }
        assert op.get_num_queued_tokens() == 150

    def test_all_non_prefilling(self):
        op = OutputProcessor.__new__(OutputProcessor)
        op.request_states = {
            "r1": _make_req_state(100, is_prefilling=False),
            "r2": _make_req_state(200, is_prefilling=False),
        }
        assert op.get_num_queued_tokens() == 0


# ---------------------------------------------------------------------------
# AsyncLLM._validate_request_scheduling
# ---------------------------------------------------------------------------


class TestValidateRequestScheduling:
    """Tests for AsyncLLM._validate_request_scheduling."""

    def test_no_limits_allows_everything(self):
        llm = _make_async_llm(num_unfinished=999, num_queued_tokens=999)
        params = SamplingParams()
        llm._validate_request_scheduling("req1", params)

    # -- max_num_queued_reqs ------------------------------------------------

    def test_reqs_allows_when_under_limit(self):
        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=5)
        params = SamplingParams()
        llm._validate_request_scheduling("req1", params)

    def test_reqs_rejects_at_limit(self):
        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
        params = SamplingParams()
        with pytest.raises(QueueOverflowError):
            llm._validate_request_scheduling("req1", params)

    def test_reqs_rejects_with_n(self):
        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=8)
        params = SamplingParams(n=3)
        with pytest.raises(QueueOverflowError):
            llm._validate_request_scheduling("req1", params)

    def test_reqs_allows_n_at_boundary(self):
        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=7)
        params = SamplingParams(n=3)
        llm._validate_request_scheduling("req1", params)

    def test_reqs_rejects_when_zero_limit(self):
        llm = _make_async_llm(max_num_queued_reqs=0, num_unfinished=0)
        params = SamplingParams()
        with pytest.raises(QueueOverflowError):
            llm._validate_request_scheduling("req1", params)

    # -- max_num_queued_tokens ----------------------------------------------

    def test_tokens_allows_when_under_limit(self):
        llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=500)
        params = SamplingParams()
        llm._validate_request_scheduling("req1", params)

    def test_tokens_rejects_at_limit(self):
        llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1000)
        params = SamplingParams()
        with pytest.raises(MaxQueuedTokensError):
            llm._validate_request_scheduling("req1", params)

    def test_tokens_rejects_over_limit(self):
        llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1500)
        params = SamplingParams()
        with pytest.raises(MaxQueuedTokensError):
            llm._validate_request_scheduling("req1", params)

    def test_tokens_rejects_when_zero_limit(self):
        llm = _make_async_llm(max_num_queued_tokens=0, num_queued_tokens=0)
        params = SamplingParams()
        with pytest.raises(MaxQueuedTokensError):
            llm._validate_request_scheduling("req1", params)

    # -- interaction between both limits -----------------------------------

    def test_both_limits_checked_independently(self):
        llm = _make_async_llm(
            max_num_queued_reqs=100,
            max_num_queued_tokens=1000,
            num_unfinished=5,
            num_queued_tokens=1000,
        )
        params = SamplingParams()
        with pytest.raises(MaxQueuedTokensError):
            llm._validate_request_scheduling("req1", params)

    def test_req_limit_checked_before_token_limit(self):
        llm = _make_async_llm(
            max_num_queued_reqs=10,
            max_num_queued_tokens=1000,
            num_unfinished=10,
            num_queued_tokens=1000,
        )
        params = SamplingParams()
        with pytest.raises(QueueOverflowError):
            llm._validate_request_scheduling("req1", params)

    # -- pooling params (no .n attribute) -----------------------------------

    def test_pooling_params_treated_as_n1(self):
        from vllm.pooling_params import PoolingParams

        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
        params = PoolingParams()
        with pytest.raises(QueueOverflowError):
            llm._validate_request_scheduling("req1", params)

    def test_pooling_params_allowed_when_under_limit(self):
        from vllm.pooling_params import PoolingParams

        llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=9)
        params = PoolingParams()
        llm._validate_request_scheduling("req1", params)


# ---------------------------------------------------------------------------
# create_error_response integration
# ---------------------------------------------------------------------------


class TestCreateErrorResponse:
    """Tests that GracefulHTTPError maps to the correct HTTP response."""

    def test_queue_overflow_maps_to_503(self):
        from vllm.entrypoints.serve.utils.error_response import (
            create_error_response,
        )

        resp = create_error_response(QueueOverflowError())
        assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
        assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
        assert resp.error.param is None
        msg = resp.error.message.lower()
        assert "busy" in msg or "try again" in msg

    def test_max_queued_tokens_maps_to_503(self):
        from vllm.entrypoints.serve.utils.error_response import (
            create_error_response,
        )

        resp = create_error_response(MaxQueuedTokensError())
        assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
        assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
        msg = resp.error.message.lower()
        assert "backlog" in msg or "try again" in msg

    def test_custom_graceful_error_maps_to_its_status(self):
        from vllm.entrypoints.serve.utils.error_response import (
            create_error_response,
        )

        err = GracefulHTTPError("custom", HTTPStatus.SERVICE_UNAVAILABLE)
        resp = create_error_response(err)
        assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
        assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase


# ---------------------------------------------------------------------------
# SchedulerConfig field defaults
# ---------------------------------------------------------------------------


class TestSchedulerConfig:
    """Tests that SchedulerConfig accepts and validates the new fields."""

    def test_defaults_are_none(self):
        from vllm.config.scheduler import SchedulerConfig

        config = SchedulerConfig(
            runner_type="generate",
            max_model_len=4096,
            is_encoder_decoder=False,
        )
        assert config.max_num_queued_reqs is None
        assert config.max_num_queued_tokens is None

    def test_accepts_explicit_values(self):
        from vllm.config.scheduler import SchedulerConfig

        config = SchedulerConfig(
            runner_type="generate",
            max_model_len=4096,
            is_encoder_decoder=False,
            max_num_queued_reqs=100,
            max_num_queued_tokens=32000,
        )
        assert config.max_num_queued_reqs == 100
        assert config.max_num_queued_tokens == 32000

    def test_rejects_negative_reqs(self):
        from pydantic import ValidationError

        from vllm.config.scheduler import SchedulerConfig

        with pytest.raises(ValidationError):
            SchedulerConfig(
                runner_type="generate",
                max_model_len=4096,
                is_encoder_decoder=False,
                max_num_queued_reqs=-1,
            )

    def test_rejects_negative_tokens(self):
        from pydantic import ValidationError

        from vllm.config.scheduler import SchedulerConfig

        with pytest.raises(ValidationError):
            SchedulerConfig(
                runner_type="generate",
                max_model_len=4096,
                is_encoder_decoder=False,
                max_num_queued_tokens=-1,
            )


# ---------------------------------------------------------------------------
# human_readable_int for CLI notation
# ---------------------------------------------------------------------------


class TestHumanReadableNotation:
    """Tests that max_num_queued_tokens accepts human-readable values."""

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
    def test_parses_notation(self, input_str: str, expected: int):
        assert human_readable_int(input_str) == expected

    @pytest.mark.parametrize("invalid", ["abc", "1x", "", "k", "1.5K"])
    def test_rejects_invalid(self, invalid: str):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            human_readable_int(invalid)
