# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu.async_utils as async_utils


def test_synchronize_event_waits_until_ready(monkeypatch):
    event = Mock()
    event.query.side_effect = [False, True]
    monotonic = iter([0.0, 0.1])
    sleep = Mock()
    monkeypatch.setattr(async_utils.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(async_utils.time, "sleep", sleep)

    async_utils._synchronize_event(
        event,
        event_name="test event",
        timeout_s=1.0,
    )

    assert event.query.call_count == 2
    sleep.assert_called_once_with(async_utils._ASYNC_OUTPUT_POLL_INTERVAL_S)


def test_synchronize_event_times_out(monkeypatch):
    event = Mock()
    event.query.return_value = False
    monotonic = iter([0.0, 1.0])
    monkeypatch.setattr(async_utils.time, "monotonic", lambda: next(monotonic))

    with pytest.raises(TimeoutError, match="test event"):
        async_utils._synchronize_event(
            event,
            event_name="test event",
            timeout_s=1.0,
        )
