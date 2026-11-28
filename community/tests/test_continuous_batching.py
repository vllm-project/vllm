"""Tests for the continuous-batching community model (sprint note 063/120)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_community import continuous_batching as model  # noqa: E402


def test_module_metadata():
    assert model.TOPIC == "continuous-batching"
    assert model.SUMMARY.startswith('Instead of batching whole requests, the ')


def test_budget_defaults_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.DEFAULTS.batch_token_budget = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("requests", [0, 1, 8, 64, 512])
def test_simulate_is_deterministic(requests):
    first = model.simulate(requests)
    second = model.simulate(requests)
    assert first == second
    assert first["requests"] == requests
    assert model.check_invariants(first) == []


def test_negative_requests_rejected():
    with pytest.raises(ValueError):
        model.simulate(-1)


def test_utilisation_is_monotonic_in_requests():
    assert model.simulate(8)["utilisation"] <= model.simulate(256)["utilisation"]


def test_doubling_primary_knob_raises_pressure():
    baseline = model.simulate(64)
    heavier = dataclasses.replace(model.DEFAULTS, batch_token_budget=model.DEFAULTS.batch_token_budget * 2.0)
    assert model.simulate(64, heavier)["pressure"] > baseline["pressure"]


def test_recommend_always_returns_advice():
    advice = model.recommend(model.simulate(256))
    assert advice
    assert all(isinstance(line, str) and line for line in advice)


def test_sweep_is_ordered_and_labelled():
    results = model.sweep(128)
    assert [item["factor"] for item in results] == [0.5, 1.0, 2.0]
    assert all(item["topic"] == model.TOPIC for item in results)
