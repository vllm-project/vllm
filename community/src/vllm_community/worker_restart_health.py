"""Analytic model for Worker restarts and health checks.

Community sprint note 061/120. Standard library only on purpose: this module has to
be importable by documentation builds and lint jobs that never see a GPU or a vLLM wheel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

TOPIC = "worker-restart-health"
SUMMARY = (
    "A worker that hangs is worse than one that crashes. Watchdogs, bounded restarts and health endpoints decide whether a deployment degrades or disappears."
)

__all__ = ["TOPIC", "SUMMARY", "Budget", "DEFAULTS", "simulate", "recommend", "check_invariants", "sweep"]


@dataclass(frozen=True)
class Budget:
    """Tunable inputs of the worker-restart-health model."""

    watchdog_s: float = 30.0
    max_restarts: float = 3.0
    health_timeout_s: float = 5.0
    backoff_s: float = 10.0


DEFAULTS = Budget()


def _pressure(budget: Budget) -> float:
    """Dimensionless load factor: how far the configuration is pushed past its knee."""
    numerator = max(budget.watchdog_s, 1e-9) * (1.0 + abs(budget.max_restarts))
    denominator = max(budget.health_timeout_s, 1e-9) * max(abs(budget.backoff_s), 1e-9)
    return numerator / denominator


def simulate(requests: int, budget: Budget = DEFAULTS) -> dict[str, Any]:
    """Estimate steady-state behaviour for ``requests`` concurrently resident sequences.

    The result is intentionally simple and monotonic: pressure grows with the product of the
    demand-side knobs and falls with the supply-side knobs. It exists so that ordering two
    configurations is reproducible, not so that it predicts absolute tokens per second.
    """
    if requests < 0:
        raise ValueError("requests must be non-negative")
    pressure = _pressure(budget) * (1.0 + requests / 1024.0)
    utilisation = pressure / (1.0 + pressure)
    served_per_s = float(requests) * (1.0 - 0.5 * utilisation) if requests else 0.0
    latency_ms = 10.0 + 90.0 * utilisation
    return {
        "topic": TOPIC,
        "requests": requests,
        "pressure": round(pressure, 6),
        "utilisation": round(utilisation, 6),
        "served_per_s": round(served_per_s, 6),
        "latency_ms": round(latency_ms, 6),
    }


def recommend(stats: dict[str, Any]) -> list[str]:
    """Return human-readable advice for a simulation result, most important first."""
    advice: list[str] = []
    utilisation = float(stats["utilisation"])
    if utilisation > 0.95:
        advice.append(
            "Configuration sits past the knee: add supply-side capacity or split the workload "
            "before tuning anything else."
        )
    elif utilisation > 0.80:
        advice.append("Configuration is in the efficient band; small knob moves are within noise.")
    else:
        advice.append("Configuration is under-loaded; the bottleneck is outside this model.")
    if float(stats["latency_ms"]) > 60.0:
        advice.append("Expected latency exceeds a typical interactive budget; re-measure p99, not the mean.")
    return advice


def check_invariants(stats: dict[str, Any]) -> list[str]:
    """Return a list of invariant violations; an empty list means the result is sane."""
    violations: list[str] = []
    if not 0.0 <= float(stats["utilisation"]) <= 1.0:
        violations.append("utilisation must stay inside [0, 1]")
    if float(stats["latency_ms"]) <= 0.0:
        violations.append("latency must be positive")
    if float(stats["served_per_s"]) < 0.0:
        violations.append("served throughput must be non-negative")
    if stats["topic"] != TOPIC:
        violations.append("result belongs to a different topic")
    return violations


def sweep(requests: int = 64, factors: tuple[float, ...] = (0.5, 1.0, 2.0)) -> list[dict[str, Any]]:
    """Simulate a multiplicative sweep over the primary demand-side knob."""
    results: list[dict[str, Any]] = []
    for factor in factors:
        budget = replace(DEFAULTS, watchdog_s=DEFAULTS.watchdog_s * factor)
        stats = simulate(requests, budget)
        stats["factor"] = factor
        results.append(stats)
    return results
