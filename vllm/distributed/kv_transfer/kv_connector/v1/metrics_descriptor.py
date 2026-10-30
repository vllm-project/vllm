# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for emitting MetricsDescriptorV1 on the KV connector stats wire.

Rust frontend cannot call ``build_prom_metrics()``. Connectors that need
descriptor-driven Prometheus series emit ``_metrics_descriptor`` once in their
``to_dict()`` payload when ``VLLM_USE_RUST_FRONTEND`` is enabled. Python
frontend scrapes leave the payload untouched.

In-tree builtins derive MetricsDescriptorV1 from the same metric definitions
used by Python Prom (or a shared declarative table), not from hand-maintained
JSON files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import vllm.envs as envs

METRICS_DESCRIPTOR_KEY = "_metrics_descriptor"
DESCRIPTOR_VERSION = 1

# Sample kinds must match Rust ``SampleKind`` (snake_case).
INC_BY_U64 = "inc_by_u64"
INC_BY_SUM_U64 = "inc_by_sum_u64"
INC_BY_F64 = "inc_by_f64"
SET_F64 = "set_f64"
OBSERVE_EACH_F64 = "observe_each_f64"

# Process-local: first to_dict per connector_id carries the descriptor.
_emitted_connector_ids: set[str] = set()


def metric_def(
    *,
    name: str,
    type: str,
    documentation: str,
    samples_path: str,
    sample_kind: str,
    buckets: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build one MetricsDescriptorV1 metric entry."""
    entry: dict[str, Any] = {
        "name": name,
        "type": type,
        "documentation": documentation,
        "samples_path": samples_path,
        "sample_kind": sample_kind,
    }
    if buckets is not None:
        entry["buckets"] = list(buckets)
    return entry


def build_metrics_descriptor(
    connector_id: str,
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a MetricsDescriptorV1 document for ``connector_id``."""
    return {
        "descriptor_version": DESCRIPTOR_VERSION,
        "connector_id": connector_id,
        "metrics": [dict(m) for m in metrics],
    }


def reset_metrics_descriptor_emission_for_tests() -> None:
    """Clear one-shot emission state (unit tests only)."""
    _emitted_connector_ids.clear()


def strip_metrics_descriptor(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop ``_metrics_descriptor`` when reconstructing stats from a wire dict."""
    if data is None or METRICS_DESCRIPTOR_KEY not in data:
        return data
    out = dict(data)
    out.pop(METRICS_DESCRIPTOR_KEY)
    return out


def maybe_attach_metrics_descriptor(
    payload: dict[str, Any],
    *,
    connector_id: str,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Attach ``_metrics_descriptor`` once when the Rust frontend is enabled.

    Args:
        payload: Connector stats dict (must not already be shared mutably
            with callers that must not see the descriptor key).
        connector_id: Class name / descriptor ``connector_id`` for one-shot tracking.
        descriptor: MetricsDescriptorV1 document.

    Returns:
        ``payload`` unchanged for Python frontend, or a shallow copy with
        ``_metrics_descriptor`` on the first Rust-frontend emit for this id.

    """
    if not envs.VLLM_USE_RUST_FRONTEND:
        return payload
    if connector_id in _emitted_connector_ids:
        return payload
    _emitted_connector_ids.add(connector_id)
    out = dict(payload)
    out[METRICS_DESCRIPTOR_KEY] = descriptor
    return out
