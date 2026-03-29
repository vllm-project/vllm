# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoggingStatLogger."""

from unittest.mock import patch

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats


def _make_logger() -> LoggingStatLogger:
    return LoggingStatLogger(vllm_config=VllmConfig())


def _make_iteration_stats_with_preemptions(n: int) -> IterationStats:
    stats = IterationStats()
    stats.num_preempted_reqs = n
    return stats


def test_preemptions_appear_in_log_when_nonzero():
    """Preemption count must be logged when preemptions occurred."""
    logger = _make_logger()
    logger._track_iteration_stats(_make_iteration_stats_with_preemptions(3))

    logged_messages = []
    with patch("vllm.v1.metrics.loggers.logger") as mock_log:
        mock_log.info.side_effect = lambda msg, *args: logged_messages.append(
            msg % args
        )
        mock_log.debug.side_effect = lambda msg, *args: logged_messages.append(
            msg % args
        )
        logger.log()

    assert any(
        "Preemptions" in msg for msg in logged_messages
    ), "Expected 'Preemptions' in log output, but it was missing"


def test_preemptions_not_in_log_when_zero():
    """Preemption count must not appear in log when no preemptions occurred."""
    logger = _make_logger()
    # No preemptions tracked

    logged_messages = []
    with patch("vllm.v1.metrics.loggers.logger") as mock_log:
        mock_log.info.side_effect = lambda msg, *args: logged_messages.append(
            msg % args
        )
        mock_log.debug.side_effect = lambda msg, *args: logged_messages.append(
            msg % args
        )
        logger.log()

    assert not any(
        "Preemptions" in msg for msg in logged_messages
    ), "Expected 'Preemptions' to be absent from log output, but it was present"
