# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoggingStatLogger."""

from unittest.mock import patch

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats


def _make_logger() -> LoggingStatLogger:
    return LoggingStatLogger(vllm_config=VllmConfig())


def _collect_log_output(logger: LoggingStatLogger) -> list[str]:
    messages: list[str] = []
    with patch("vllm.v1.metrics.loggers.logger") as mock_log:
        mock_log.info.side_effect = lambda msg, *args: messages.append(
            msg % args
        )
        mock_log.debug.side_effect = lambda msg, *args: messages.append(
            msg % args
        )
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
    """Preemption count must be logged when preemptions occurred.

    This test would have failed before the fix: _update_stats() called _reset()
    which zeroed num_preemptions before the > 0 guard was checked.
    """
    logger = _make_logger()
    logger._track_iteration_stats(_make_iteration_stats_with_preemptions(3))

    messages = _collect_log_output(logger)

    assert any(
        "Preemptions: 3" in msg for msg in messages
    ), "Expected 'Preemptions: 3' in log output, but it was missing"


def test_preemptions_not_in_log_when_zero():
    """Preemption count must not appear in log when no preemptions occurred."""
    logger = _make_logger()

    messages = _collect_log_output(logger)

    assert not any(
        "Preemptions" in msg for msg in messages
    ), "Expected 'Preemptions' to be absent from log output, but it was present"


def test_corrupted_reqs_value_correct_in_log():
    """Corrupted req count must reflect actual count, not post-reset zero.

    This test would have failed before the fix: _update_stats() called _reset()
    which zeroed num_corrupted_reqs before it was read for logging.
    """
    logger = _make_logger()
    logger._track_iteration_stats(_make_iteration_stats_with_corrupted(5))

    with patch("vllm.v1.metrics.loggers.envs") as mock_envs:
        mock_envs.VLLM_COMPUTE_NANS_IN_LOGITS = True
        messages = _collect_log_output(logger)

    assert any(
        "Corrupted: 5 reqs" in msg for msg in messages
    ), "Expected 'Corrupted: 5 reqs' in log output, but it was missing"
