import pytest
# ABOUTME: Tests EPS reporter and aggregator accumulation logic.
# ABOUTME: Ensures snapshot ratios and averages are computed correctly.

from vllm.v1.eps.reporter import EpsAggregator
from vllm.v1.eps.telemetry import EpsStepCounters


def test_eps_aggregator_snapshot():
    agg = EpsAggregator()
    agg.ingest(EpsStepCounters(
        pages_total=10,
        pages_visited=6,
        pages_skipped=4,
        kv_bytes_total=1000,
        kv_bytes_kept=400,
        eps_prepass_ms=2.0,
        decode_ms=8.0,
    ))
    agg.ingest(EpsStepCounters(
        pages_total=10,
        pages_visited=5,
        pages_skipped=5,
        kv_bytes_total=1000,
        kv_bytes_kept=600,
        eps_prepass_ms=4.0,
        decode_ms=12.0,
    ))

    snapshot = agg.snapshot()
    assert snapshot["eps.pages_kept_ratio"] == pytest.approx(0.55)
    assert snapshot["eps.pages_skipped_ratio"] == pytest.approx(0.45)
    assert snapshot["eps.bytes_kept_ratio"] == pytest.approx(0.5)
    assert snapshot["eps.prepass_ms_avg"] == pytest.approx(3.0)
    assert snapshot["eps.decode_ms_avg"] == pytest.approx(10.0)
