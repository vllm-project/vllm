# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias, cast

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)

ExternalMetricValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["ExternalMetricValue"]
    | dict[str, "ExternalMetricValue"]
)
ExternalMetricsPayload: TypeAlias = Mapping[str, ExternalMetricValue]
ExternalMetricsSnapshot: TypeAlias = dict[str, dict[str, ExternalMetricValue]]
ExternalMetricsProvider: TypeAlias = Callable[[], ExternalMetricsPayload | None]

_PROVIDER_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DEFAULT_COLLECTION_INTERVAL_S = 1.0


@dataclass
class _ProviderState:
    provider: ExternalMetricsProvider
    collection_interval_s: float
    next_collection_time: float = 0.0


_providers: dict[str, _ProviderState] = {}
_providers_lock = threading.Lock()


def _normalize_metric_value(value: object) -> ExternalMetricValue:
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("External metrics float values must be finite.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_metric_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, ExternalMetricValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("External metrics mapping keys must be strings.")
            normalized[key] = _normalize_metric_value(item)
        return normalized
    raise TypeError(
        "External metrics values must be JSON-serializable scalar, list, "
        "or mapping values."
    )


def register_external_metrics_provider(
    name: str,
    provider: ExternalMetricsProvider,
    *,
    collection_interval_s: float = _DEFAULT_COLLECTION_INTERVAL_S,
) -> None:
    """Register a process-local external metrics provider.

    General plugins may call this function in every vLLM process during plugin
    initialization, before the engine core constructs its scheduler. Providers
    are collected only by the engine core and their snapshots are delivered to
    stat logger plugins through ``SchedulerStats.external_metrics``.

    Providers run on the engine-core thread and must return promptly. The
    collection interval keeps provider work off the per-step hot path.
    """
    if not _PROVIDER_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "External metrics provider names must start with a letter and "
            "contain only letters, digits, '.', '_', or '-'."
        )
    if not callable(provider):
        raise TypeError("External metrics provider must be callable.")
    if not isfinite(collection_interval_s) or collection_interval_s <= 0:
        raise ValueError(
            "collection_interval_s must be a finite value greater than zero."
        )

    with _providers_lock:
        if name in _providers:
            raise ValueError(
                f"External metrics provider {name!r} is already registered."
            )
        _providers[name] = _ProviderState(provider, collection_interval_s)


def unregister_external_metrics_provider(name: str) -> None:
    """Unregister an external metrics provider if it is present."""
    with _providers_lock:
        _providers.pop(name, None)


def has_external_metrics_providers() -> bool:
    """Return whether any provider was registered in this process."""
    with _providers_lock:
        return bool(_providers)


def collect_external_metrics(
    *, now: float | None = None
) -> ExternalMetricsSnapshot | None:
    """Collect providers whose configured interval has elapsed.

    Provider failures are isolated from the engine. ``None`` means that no
    provider produced an update during this call; consumers should retain the
    last snapshot they received for each provider.
    """
    collection_time = time.monotonic() if now is None else now
    due_providers: list[tuple[str, ExternalMetricsProvider]] = []

    with _providers_lock:
        for name, state in _providers.items():
            if collection_time < state.next_collection_time:
                continue
            state.next_collection_time = collection_time + state.collection_interval_s
            due_providers.append((name, state.provider))

    snapshots: ExternalMetricsSnapshot = {}
    for name, provider in due_providers:
        try:
            payload = provider()
            if payload is not None:
                if not isinstance(payload, Mapping):
                    raise TypeError("External metrics providers must return a mapping.")
                snapshots[name] = cast(
                    dict[str, ExternalMetricValue],
                    _normalize_metric_value(payload),
                )
        except Exception:
            logger.exception("External metrics provider %r failed", name)

    return snapshots or None


def _reset_external_metrics_providers_for_tests() -> None:
    with _providers_lock:
        _providers.clear()
