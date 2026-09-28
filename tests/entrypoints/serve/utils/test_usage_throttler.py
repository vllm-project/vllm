# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the shared continuous usage-stats throttler.

``ContinuousUsageStatsThrottler`` lives in ``api_utils`` next to
``should_include_usage`` — the shared usage-stats streaming policy — and
defines the emission cadence for streaming endpoints whose protocol
conveys intermediate usage as discrete events (``/v1/messages``
``message_delta``). These tests pin down that contract without a server.
"""

import pytest

from vllm.entrypoints.serve.utils.api_utils import (
    CONTINUOUS_USAGE_EMISSION_TOKEN_INTERVAL,
    ContinuousUsageStatsThrottler,
)


class TestContinuousUsageStatsThrottler:
    def test_first_update_emitted_eagerly(self):
        throttler = ContinuousUsageStatsThrottler()
        assert throttler.should_emit(1) is True

    def test_subsequent_updates_below_interval_are_suppressed(self):
        throttler = ContinuousUsageStatsThrottler()
        throttler.should_emit(1)
        for tokens in range(2, CONTINUOUS_USAGE_EMISSION_TOKEN_INTERVAL):
            assert throttler.should_emit(tokens) is False

    def test_update_emitted_once_interval_elapsed(self):
        throttler = ContinuousUsageStatsThrottler(interval=50)
        throttler.should_emit(10)
        assert throttler.should_emit(59) is False
        assert throttler.should_emit(60) is True  # 60 - 10 >= 50
        assert throttler.should_emit(61) is False

    def test_interval_counts_from_last_emission(self):
        throttler = ContinuousUsageStatsThrottler(interval=50)
        throttler.should_emit(0)
        assert throttler.should_emit(50) is True
        assert throttler.should_emit(99) is False
        assert throttler.should_emit(100) is True

    def test_terminal_always_suppressed(self):
        throttler = ContinuousUsageStatsThrottler()
        # Even the very first update is dropped once the stream is terminal:
        # only the true final usage summary should carry the last counts.
        assert throttler.should_emit(120, terminal=True) is False

    def test_terminal_suppresses_after_partial_emissions(self):
        throttler = ContinuousUsageStatsThrottler(interval=50)
        throttler.should_emit(10)
        throttler.should_emit(60)
        assert throttler.should_emit(61, terminal=True) is False

    @pytest.mark.parametrize("interval", [1, 10, 50, 1000])
    def test_monotonic_counts_never_regress(self, interval):
        """Simulate a full stream; the throttler must never emit twice for
        the same token count and never emit after the terminal state."""
        throttler = ContinuousUsageStatsThrottler(interval=interval)
        emitted: list[int] = []
        terminal = False
        for tokens in range(0, 500):
            if tokens == 480:
                terminal = True
            if throttler.should_emit(tokens, terminal=terminal):
                emitted.append(tokens)
        assert len(emitted) >= 1
        assert emitted == sorted(set(emitted))
        # No emission at or after the terminal boundary.
        assert all(t < 480 for t in emitted)
