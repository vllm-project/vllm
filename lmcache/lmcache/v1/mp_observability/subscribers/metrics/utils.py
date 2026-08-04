# SPDX-License-Identifier: Apache-2.0

"""Shared cache_salt helpers for metrics subscribers."""

# Future
from __future__ import annotations

# Standard
from collections import Counter

# Third Party
from opentelemetry import metrics


def group_by_salt(keys: list) -> dict[str, int]:
    """Group *keys* by ``cache_salt``, returning salt → count."""
    return Counter(getattr(k, "cache_salt", "") for k in keys)


def emit_salt_counts(counter: metrics.Counter, salt_counts: dict[str, int]) -> None:
    """Add to *counter* once per entry in *salt_counts*.

    *salt_counts* maps ``cache_salt`` values to key counts.
    Empty salt produces a dimensionless increment (no attribute).
    """
    for salt, count in salt_counts.items():
        attrs = {"cache_salt": salt} if salt else {}
        counter.add(count, attributes=attrs)
