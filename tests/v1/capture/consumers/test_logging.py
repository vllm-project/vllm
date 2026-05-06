# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``LoggingConsumer`` reference consumer."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.capture.consumers.logging import LoggingConsumer
from vllm.v1.capture.types import (
    CaptureKey,
    CaptureSpec,
    VllmInternalRequestId,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_CONFIG = MagicMock()

_LOGGER_NAME = "vllm.capture.logging"


def _key(req_id: str = "req-1", layer: int = 0, hook: str = "post_mlp") -> CaptureKey:
    return (VllmInternalRequestId(req_id), layer, hook)


@pytest.fixture(autouse=True)
def _enable_propagation():
    """Ensure the vllm.capture.logging logger propagates all the way
    to the root so that pytest's ``caplog`` fixture can capture records.

    vLLM's logging config sets ``propagate=False`` on the ``vllm``
    logger, which blocks caplog from seeing child records. We
    temporarily override propagation on both the target logger and
    the ``vllm`` parent.
    """
    target = logging.getLogger(_LOGGER_NAME)
    vllm_logger = logging.getLogger("vllm")
    orig_target = target.propagate
    orig_vllm = vllm_logger.propagate
    target.propagate = True
    vllm_logger.propagate = True
    yield
    target.propagate = orig_target
    vllm_logger.propagate = orig_vllm


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction():
    """LoggingConsumer constructs without error when given valid params."""
    consumer = LoggingConsumer(
        _MOCK_CONFIG,
        {"hooks": {"post_mlp": [0]}, "positions": "last_prompt"},
    )
    assert consumer is not None


# ---------------------------------------------------------------------------
# global_capture_spec
# ---------------------------------------------------------------------------


def test_global_capture_spec_returns_configured_spec():
    """global_capture_spec returns a CaptureSpec with the configured hooks
    and positions."""
    consumer = LoggingConsumer(
        _MOCK_CONFIG,
        {"hooks": {"post_mlp": [0, 1], "pre_attn": [2]}, "positions": "all"},
    )
    spec = consumer.global_capture_spec()
    assert isinstance(spec, CaptureSpec)
    assert spec.hooks == {"post_mlp": [0, 1], "pre_attn": [2]}
    assert spec.positions == "all"


# ---------------------------------------------------------------------------
# on_capture logging
# ---------------------------------------------------------------------------


def test_on_capture_logs_key_rows_dtype(caplog: pytest.LogCaptureFixture):
    """on_capture emits a log message containing the key, row count, and
    dtype."""
    consumer = LoggingConsumer(
        _MOCK_CONFIG,
        {"hooks": {"post_mlp": [0]}},
    )
    key = _key()
    tensor = torch.randn(5, 16)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        consumer.on_capture(key, tensor, {})

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert str(key) in record.message
    assert "rows=5" in record.message
    assert str(tensor.dtype) in record.message


# ---------------------------------------------------------------------------
# Custom level
# ---------------------------------------------------------------------------


def test_custom_level_debug(caplog: pytest.LogCaptureFixture):
    """Construct with level='DEBUG', verify log message at DEBUG level."""
    consumer = LoggingConsumer(
        _MOCK_CONFIG,
        {"hooks": {"post_mlp": [0]}, "level": "DEBUG"},
    )
    key = _key()
    tensor = torch.randn(3, 8)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        consumer.on_capture(key, tensor, {})

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


# ---------------------------------------------------------------------------
# Default positions
# ---------------------------------------------------------------------------


def test_default_positions():
    """Construct without positions param, verify default is 'last_prompt'."""
    consumer = LoggingConsumer(
        _MOCK_CONFIG,
        {"hooks": {"post_mlp": [0]}},
    )
    spec = consumer.global_capture_spec()
    assert spec.positions == "last_prompt"
