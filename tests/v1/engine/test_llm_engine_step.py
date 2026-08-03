# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.stats import SchedulerIterationDetails


def test_step_with_stats_returns_existing_iteration_details(monkeypatch):
    engine = object.__new__(LLMEngine)
    details = SchedulerIterationDetails(
        iteration_index=4,
        num_ctx_requests=1,
        num_ctx_tokens=512,
        num_generation_requests=8,
        num_generation_tokens=8,
        elapsed_ms=3.5,
    )
    monkeypatch.setattr(engine, "has_unfinished_requests", lambda: True)

    def step():
        engine._last_scheduler_stats = SimpleNamespace(
            iteration_details=details, kv_cache_usage=0.25
        )
        return []

    monkeypatch.setattr(engine, "step", step)
    result = engine.step_with_stats()

    assert result.outputs == []
    assert result.stats is details
    assert result.kv_cache_usage == 0.25


def test_step_with_stats_skips_frontend_only_outputs(monkeypatch):
    engine = object.__new__(LLMEngine)
    details = SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=1,
        num_generation_tokens=1,
        elapsed_ms=1.0,
    )
    monkeypatch.setattr(engine, "has_unfinished_requests", lambda: True)
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        engine._last_scheduler_stats = (
            None
            if calls == 1
            else SimpleNamespace(iteration_details=details, kv_cache_usage=0.5)
        )
        return [f"output-{calls}"]

    monkeypatch.setattr(engine, "step", step)
    result = engine.step_with_stats()

    assert result.outputs == ["output-1", "output-2"]
    assert result.stats is details


def test_step_with_stats_requires_iteration_details(monkeypatch):
    engine = object.__new__(LLMEngine)
    monkeypatch.setattr(engine, "has_unfinished_requests", lambda: True)

    def step():
        engine._last_scheduler_stats = SimpleNamespace(
            iteration_details=None, kv_cache_usage=0.0
        )
        return []

    monkeypatch.setattr(engine, "step", step)
    with pytest.raises(RuntimeError, match="iteration details are disabled"):
        engine.step_with_stats()


def test_step_with_stats_requires_model_iteration(monkeypatch):
    engine = object.__new__(LLMEngine)
    monkeypatch.setattr(engine, "has_unfinished_requests", lambda: False)
    with pytest.raises(RuntimeError, match="no model iteration"):
        engine.step_with_stats()
