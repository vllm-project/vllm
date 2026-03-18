# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AdaptiveOffloadingPolicy (Strategy B / P2).

The production class is loaded by extracting its AST node from
offloading_connector.py and exec()'ing only that class definition.
This means:
  - Tests always run against the REAL production source code.
  - Any change to AdaptiveOffloadingPolicy is immediately detected.
  - No vllm import chain needed (avoids Python 3.10+ dependency in CI for
    pure-unit tests).
"""
from __future__ import annotations

import ast
import logging as _logging
import pathlib
import statistics
import textwrap
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

import pytest

# Structural stub used by mypy only — describes the interface of the
# dynamically-loaded AdaptiveOffloadingPolicy without requiring a full vllm
# import at type-check time.
if TYPE_CHECKING:
    class AdaptiveOffloadingPolicy(Protocol):  # pragma: no cover
        paused: bool
        baseline_ttft: float | None
        overhead_threshold_pct: float

        def __init__(
            self,
            overhead_threshold_pct: float = ...,
            window: int = ...,
            warmup_steps: int = ...,
            expected_baseline_ttft_ms: float | None = ...,
        ) -> None: ...

        def record_ttft(self, ttft_ms: float) -> None: ...

        @property
        def effective_load_mode(self) -> str: ...

# ---------------------------------------------------------------------------
# Load the real AdaptiveOffloadingPolicy from production source
# ---------------------------------------------------------------------------

_CONNECTOR_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"
)


def _extract_class_source(path: pathlib.Path, class_name: str) -> str:
    """Parse the file's AST and return the exact source lines of class_name."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # ast gives 1-indexed line numbers
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise RuntimeError(f"Class {class_name!r} not found in {path}")


# Exec the class definition into an isolated namespace.
# The class needs: statistics, deque (stdlib), and logger (module-level global).
_ns: dict[str, Any] = {
    "statistics": statistics,
    "deque": deque,
    "logger": _logging.getLogger("test.adaptive_policy"),
}
exec(  # noqa: S102
    textwrap.dedent(_extract_class_source(_CONNECTOR_PATH, "AdaptiveOffloadingPolicy")),
    _ns,
)
if not TYPE_CHECKING:
    AdaptiveOffloadingPolicy = _ns["AdaptiveOffloadingPolicy"]  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_policy(**kwargs) -> "AdaptiveOffloadingPolicy":
    defaults = dict(overhead_threshold_pct=5.0, window=20, warmup_steps=3)
    defaults.update(kwargs)
    return AdaptiveOffloadingPolicy(**defaults)


# ---------------------------------------------------------------------------
# Tests: warm-up behaviour
# ---------------------------------------------------------------------------

class TestAdaptiveOffloadingPolicyWarmup:
    def test_starts_paused(self):
        p = make_policy(warmup_steps=5)
        assert p.paused is True
        assert p.effective_load_mode == "blocking"

    def test_activates_after_warmup(self):
        p = make_policy(warmup_steps=5)
        for _ in range(5):
            p.record_ttft(10.0)
        assert p.paused is False
        assert p.effective_load_mode == "async_with_fallback"
        assert p.baseline_ttft == pytest.approx(10.0)

    def test_skips_warmup_with_provided_baseline(self):
        p = make_policy(expected_baseline_ttft_ms=12.0, warmup_steps=50)
        assert p.paused is False
        assert p.baseline_ttft == pytest.approx(12.0)

    def test_baseline_uncontaminated(self):
        """Baseline = median of warm-up TTFTs (measured while paused)."""
        p = make_policy(warmup_steps=3, window=10)
        p.record_ttft(8.0)
        p.record_ttft(10.0)
        p.record_ttft(12.0)  # baseline = median(8, 10, 12) = 10
        assert p.baseline_ttft == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: regression detection and auto-resume
# ---------------------------------------------------------------------------

class TestAdaptiveOffloadingPolicyRegression:
    def _activated(self, baseline: float = 10.0, **kwargs):
        p = make_policy(warmup_steps=3, window=20, **kwargs)
        for _ in range(3):
            p.record_ttft(baseline)
        assert p.paused is False, "policy should be active after warmup"
        return p

    def test_pauses_on_regression(self):
        p = self._activated(baseline=10.0, overhead_threshold_pct=5.0)
        for _ in range(10):
            p.record_ttft(20.0)  # 100% overhead
        assert p.paused is True

    def test_resumes_when_regression_clears(self):
        # window=20; need 20 good samples to fully flush the regression samples.
        p = self._activated(baseline=10.0, overhead_threshold_pct=5.0)
        for _ in range(10):
            p.record_ttft(20.0)
        assert p.paused is True
        for _ in range(20):
            p.record_ttft(10.0)
        assert p.paused is False

    def test_no_pause_within_threshold(self):
        p = self._activated(baseline=10.0, overhead_threshold_pct=20.0)
        for _ in range(10):
            p.record_ttft(11.0)  # 10% < 20% threshold
        assert p.paused is False


# ---------------------------------------------------------------------------
# Tests: effective_load_mode property
# ---------------------------------------------------------------------------

class TestAdaptiveOffloadingPolicyEffectiveLoadMode:
    def test_blocking_when_paused(self):
        p = make_policy(warmup_steps=10)
        assert p.effective_load_mode == "blocking"

    def test_async_after_warmup(self):
        p = make_policy(warmup_steps=2)
        p.record_ttft(10.0)
        p.record_ttft(10.0)
        assert p.effective_load_mode == "async_with_fallback"
