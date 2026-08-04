# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for tagged-counter subscriber tests.

Reads back counter data points from the in-memory OTel reader (see
``otel_setup.py``) keyed by ``(metric_name, frozenset_of_attribute_pairs)``.
This enables tests to assert on exact counts for specific attribute
combinations (e.g. ``during=l1_store``, ``model_name=llama-7b``) without
hand-rolling the plumbing in every test file.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.distributed.api import ObjectKey
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

TaggedKey = tuple[str, frozenset[tuple[str, str]]]


def make_key(model: str, chunk_id: int) -> ObjectKey:
    """Build a test ObjectKey with the given model name and chunk id."""
    return ObjectKey(
        chunk_hash=chunk_id.to_bytes(4, byteorder="big"),
        model_name=model,
        kv_rank=0,
    )


def read_tagged_counters() -> dict[TaggedKey, int]:
    """Snapshot all counter values keyed by (metric_name, attrs).

    Works for tagged counters where multiple data points differ only by
    their attribute set. Histogram data points (which lack ``value``) are
    skipped.
    """
    data = _reader.get_metrics_data()
    result: dict[TaggedKey, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    attrs = frozenset(
                        (str(k), str(v)) for k, v in dict(dp.attributes).items()
                    )
                    result[(metric.name, attrs)] = int(dp.value)
    return result


def counter_delta(
    before: dict[TaggedKey, int],
    after: dict[TaggedKey, int],
) -> dict[TaggedKey, int]:
    """Return the per-tagged-key delta between two snapshots."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


def counter_value(
    delta: dict[TaggedKey, int],
    metric: str,
    **attrs: str,
) -> int:
    """Look up a counter delta by metric name + attribute equality.

    Returns 0 if no data point matches the attribute set exactly.
    """
    key = (metric, frozenset(attrs.items()))
    return delta.get(key, 0)
