# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.v1.worker.gpu import event_utils


class FakeEvent:
    def __init__(self, query_results: list[bool]):
        self.query_results = iter(query_results)
        self.synchronize_calls = 0

    def query(self) -> bool:
        return next(self.query_results)

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def test_wait_for_gpu_event_returns_immediately(monkeypatch: pytest.MonkeyPatch):
    event = FakeEvent([True])
    monkeypatch.setattr(envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 60)
    monkeypatch.setattr(
        event_utils.time,
        "sleep",
        lambda _: pytest.fail("sleep should not be called"),
    )

    event_utils.wait_for_gpu_event(event, "test output copy")

    assert event.synchronize_calls == 0


def test_wait_for_gpu_event_polls_with_backoff(monkeypatch: pytest.MonkeyPatch):
    event = FakeEvent([False, False, True])
    sleep_calls: list[float] = []
    monkeypatch.setattr(envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 60)
    monkeypatch.setattr(event_utils.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(event_utils.time, "sleep", sleep_calls.append)

    event_utils.wait_for_gpu_event(event, "test output copy")

    assert sleep_calls == pytest.approx([0.0001, 0.0002])
    assert event.synchronize_calls == 0


def test_wait_for_gpu_event_times_out(monkeypatch: pytest.MonkeyPatch):
    event = FakeEvent([False, False, False])
    now = 0.0

    def sleep(duration: float) -> None:
        nonlocal now
        now += duration

    monkeypatch.setattr(envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 0.00025)
    monkeypatch.setattr(event_utils.time, "monotonic", lambda: now)
    monkeypatch.setattr(event_utils.time, "sleep", sleep)

    with pytest.raises(
        TimeoutError,
        match=(
            "Timed out after 0.00025s waiting for test output copy.*"
            "VLLM_ENGINE_ITERATION_TIMEOUT_S"
        ),
    ):
        event_utils.wait_for_gpu_event(event, "test output copy")

    assert event.synchronize_calls == 0


def test_wait_for_gpu_event_can_disable_timeout(monkeypatch: pytest.MonkeyPatch):
    event = FakeEvent([])
    monkeypatch.setattr(envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 0)

    event_utils.wait_for_gpu_event(event, "test output copy")

    assert event.synchronize_calls == 1
