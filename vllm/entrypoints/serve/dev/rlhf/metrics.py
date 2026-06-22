# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the RL weight-update protocol.

All metrics are registered in the default prometheus_client registry, which
means they automatically appear in the ``/metrics`` endpoint served by the
vLLM API server when ``VLLM_SERVER_DEV_MODE=1``.

Metrics
-------
vllm:rl_weight_update_total
    Counter.  Increments by 1 on every successful ``finish_weight_update``.
    Labels: ``engine`` (the engine index, always "0" for single-engine).

vllm:rl_weight_update_duration_seconds
    Histogram.  Records the wall-clock seconds from ``start_weight_update``
    to ``finish_weight_update``.  Buckets cover the expected range of 0.1 s
    to 600 s (10 min) for dense-model NCCL transfers.
    Labels: ``engine``.

vllm:rl_weight_gen
    Gauge.  Set to the current ``weight_gen`` value after every
    ``finish_weight_update``.  Useful for off-policy staleness detection:
    ``current_step_weight_gen - sample_weight_gen`` gives the staleness.
    Labels: ``engine``.

vllm:rl_weight_update_active
    Gauge (0 or 1).  Set to 1 by ``start_weight_update``, back to 0 by
    ``finish_weight_update``.  Lets operators detect hung updates.
    Labels: ``engine``.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_ENGINE_LABEL = "engine"

rl_weight_update_total = Counter(
    "vllm:rl_weight_update_total",
    "Number of completed weight update cycles (finish_weight_update calls).",
    labelnames=[_ENGINE_LABEL],
)

rl_weight_update_duration_seconds = Histogram(
    "vllm:rl_weight_update_duration_seconds",
    "Wall-clock seconds from start_weight_update to finish_weight_update.",
    labelnames=[_ENGINE_LABEL],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)

rl_weight_gen = Gauge(
    "vllm:rl_weight_gen",
    "Current weight generation counter (auto-incremented by finish_weight_update).",
    labelnames=[_ENGINE_LABEL],
)

rl_weight_update_active = Gauge(
    "vllm:rl_weight_update_active",
    "1 while a weight update is in progress (between start and finish), 0 otherwise.",
    labelnames=[_ENGINE_LABEL],
)
