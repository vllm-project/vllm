# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from collections.abc import Iterable
from threading import Lock

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, multiprocess
from prometheus_client.core import GaugeMetricFamily, Metric

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global temporary directory for prometheus multiprocessing
_prometheus_multiproc_dir: tempfile.TemporaryDirectory | None = None
_initial_gauge_values: dict[
    str, dict[tuple[tuple[str, str], ...], GaugeMetricFamily]
] = {}
_initial_gauge_values_lock = Lock()


class _MultiProcessCollectorWithInitialValues(multiprocess.MultiProcessCollector):
    def collect(self) -> Iterable[Metric]:
        with _initial_gauge_values_lock:
            defaults = {
                name: tuple(values.values())
                for name, values in _initial_gauge_values.items()
            }
        for metric in super().collect():
            initial_values = defaults.pop(metric.name, ())
            if initial_values:
                labels = {
                    tuple(sorted(sample.labels.items())) for sample in metric.samples
                }
                for initial in initial_values:
                    sample = initial.samples[0]
                    if tuple(sorted(sample.labels.items())) not in labels:
                        metric.samples.append(sample)
            yield metric
        for initial_values in defaults.values():
            first = initial_values[0]
            metric = GaugeMetricFamily(first.name, first.documentation, unit=first.unit)
            metric.samples = [initial.samples[0] for initial in initial_values]
            yield metric


def set_gauge_initial_value(gauge: Gauge, value: float) -> None:
    """Initialize a gauge without replacing real multiprocess mostrecent samples.

    In multiprocess mostrecent modes, defaults are only exposed through
    get_prometheus_registry(), not a plain MultiProcessCollector.
    Multiprocess all/liveall gauges already expose zero on creation, so a
    zero default leaves their storage unchanged.

    Args:
        gauge: An unlabeled gauge or a child with all labels bound.
        value: Startup value.

    """
    storage = getattr(gauge, "_value", None)
    mode = getattr(gauge, "_multiprocess_mode", None)
    is_multiprocess = getattr(storage, "_multiprocess", False)
    if is_multiprocess and mode in ("all", "liveall") and value == 0:
        return
    if is_multiprocess and mode in (
        "mostrecent",
        "livemostrecent",
    ):
        # Defaults belong in the scrape, not the mmap that real writers update.
        for metric in gauge.collect():
            for sample in metric.samples:
                sample_labels = dict(zip(gauge._labelnames, gauge._labelvalues))
                sample_labels.update(sample.labels)
                initial = GaugeMetricFamily(
                    metric.name,
                    metric.documentation,
                    labels=list(sample_labels),
                    unit=metric.unit,
                )
                initial.add_metric(list(sample_labels.values()), float(value))
                labels = tuple(sorted(sample_labels.items()))
                with _initial_gauge_values_lock:
                    _initial_gauge_values.setdefault(metric.name, {})[labels] = initial
        return
    gauge.set(value)


def setup_multiprocess_prometheus():
    """Set up prometheus multiprocessing directory if not already configured."""
    global _prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        # Make TemporaryDirectory for prometheus multiprocessing
        # Note: global TemporaryDirectory will be automatically
        # cleaned up upon exit.
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
        logger.debug(
            "Created PROMETHEUS_MULTIPROC_DIR at %s", _prometheus_multiproc_dir.name
        )
    else:
        logger.warning(
            "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
            "This directory must be wiped between vLLM runs or "
            "you will find inaccurate metrics. Unset the variable "
            "and vLLM will properly handle cleanup."
        )


def get_prometheus_registry() -> CollectorRegistry:
    """Get the appropriate prometheus registry based on multiprocessing
    configuration.

    Returns:
        Registry: A prometheus registry

    """
    if os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None:
        logger.debug("Using multiprocess registry for prometheus metrics")
        registry = CollectorRegistry()
        _MultiProcessCollectorWithInitialValues(registry)
        return registry

    return REGISTRY


def unregister_vllm_metrics():
    """Unregister any existing vLLM collectors from the prometheus registry.

    This is useful for testing and CI/CD where metrics may be registered
    multiple times across test runs.

    Also, in case of multiprocess, we need to unregister the metrics from the
    global registry.
    """
    registry = REGISTRY
    # Unregister any existing vLLM collectors
    for collector in list(registry._collector_to_names):
        if hasattr(collector, "_name") and collector._name.startswith("vllm:"):
            registry.unregister(collector)
    with _initial_gauge_values_lock:
        for name in list(_initial_gauge_values):
            if name.startswith("vllm:"):
                del _initial_gauge_values[name]


def shutdown_prometheus():
    """Shutdown prometheus metrics."""
    path = _prometheus_multiproc_dir
    if path is None:
        return
    try:
        pid = os.getpid()
        multiprocess.mark_process_dead(pid, path)
        logger.debug("Marked Prometheus metrics for process %d as dead", pid)
    except Exception as e:
        logger.error("Error during metrics cleanup: %s", str(e))
