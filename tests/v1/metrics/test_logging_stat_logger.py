# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoggingStatLogger."""

from unittest.mock import patch

from vllm.config import DeviceConfig, VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats


def _make_logger() -> LoggingStatLogger:
    return LoggingStatLogger(
        vllm_config=VllmConfig(device_config=DeviceConfig(device="cpu"))
    )


def _collect_log_output(logger: LoggingStatLogger) -> list[str]:
    messages: list[str] = []
    with patch("vllm.v1.metrics.loggers.logger") as mock_log:
        mock_log.info.side_effect = lambda msg, *args: messages.append(msg % args)
        mock_log.debug.side_effect = lambda msg, *args: messages.append(msg % args)
        logger.log()
    return messages


def _make_iteration_stats_with_preemptions(n: int) -> IterationStats:
    stats = IterationStats()
    stats.num_preempted_reqs = n
    return stats


def _make_iteration_stats_with_corrupted(n: int) -> IterationStats:
    stats = IterationStats()
    stats.num_corrupted_reqs = n
    return stats


def test_preemptions_appear_in_log_when_nonzero():
    logger = _make_logger()
    logger.record(None, _make_iteration_stats_with_preemptions(3))

    logged_output = "\n".join(_collect_log_output(logger))

    assert "Preemptions: 3" in logged_output


def test_preemptions_not_in_log_when_zero():
    logger = _make_logger()

    logged_output = "\n".join(_collect_log_output(logger))

    assert "Preemptions:" not in logged_output


def test_corrupted_reqs_value_correct_in_log():
    logger = _make_logger()
    logger.record(None, _make_iteration_stats_with_corrupted(5))

    with patch("vllm.v1.metrics.loggers.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True):
        logged_output = "\n".join(_collect_log_output(logger))

    assert "Corrupted: 5 reqs" in logged_output
