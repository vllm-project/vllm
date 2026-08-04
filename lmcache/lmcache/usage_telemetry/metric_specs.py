# SPDX-License-Identifier: Apache-2.0
"""Map-reduce metric definitions for MP continuous usage reporting.

:class:`MetricSpec` is the declarative unit: it maps one EventBus event
to a numeric sample and reduces the samples buffered in a flush interval
to one ``ContinuousContextMessage`` field. :func:`default_metric_specs`
is the registry of metrics sent today; adding a metric means adding a
spec here and its field to the message schema in ``messages.py``.

Like :mod:`.mp_continuous`, this module is not re-exported from the
package root: it imports :mod:`lmcache.v1.mp_observability`, which the
single-process engine path must not pull in.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Callable, Sequence

# First Party
from lmcache.v1.mp_observability.event import Event, EventType


@dataclass(frozen=True)
class MetricSpec:
    """Map-reduce definition of one continuous usage metric.

    Attributes:
        event_type: The EventBus event the metric is sampled from.
        field: The ``ContinuousContextMessage`` field receiving the
            reduced value. The reduced value is cast to ``int``.
        extract: Map step — turns one event into a numeric sample, or
            ``None`` to skip the event. May rely on the event metadata
            keys documented in ``docs/design/v1/mp_observability/EVENTS.md``.
        reduce: Reduce step — folds all samples buffered in one flush
            interval into the field value. Must accept an empty sequence
            (idle intervals are flushed as heartbeats); ``sum`` is the
            common case.
    """

    event_type: EventType
    field: str
    extract: Callable[[Event], int | float | None]
    reduce: Callable[[Sequence[int | float]], int | float]


def default_metric_specs(chunk_size: int) -> list[MetricSpec]:
    """Build the parity metrics matching the single-process reporter.

    Args:
        chunk_size: The server chunk size in tokens; converts the chunk
            counts carried by store/retrieve events to tokens.

    Returns:
        Specs covering every metric field of ``ContinuousContextMessage``.
    """
    return [
        MetricSpec(
            event_type=EventType.MP_RETRIEVE_END,
            field="interval_num_hit_tokens",
            extract=lambda e: int(e.metadata["retrieved_count"]) * chunk_size,
            reduce=sum,
        ),
        MetricSpec(
            event_type=EventType.MP_STORE_END,
            field="interval_num_stored_tokens",
            extract=lambda e: int(e.metadata["stored_count"]) * chunk_size,
            reduce=sum,
        ),
        MetricSpec(
            event_type=EventType.MP_STORE_END,
            field="interval_stored_kv_size",
            extract=lambda e: int(e.metadata["total_bytes"]),
            reduce=sum,
        ),
    ]
