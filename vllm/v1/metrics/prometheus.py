# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, multiprocess

from vllm.logger import init_logger

logger = init_logger(__name__)

# server draining gauge - set to 1 when server is shutting down gracefully
_server_draining_gauge: Gauge | None = None


def get_server_draining_gauge(model_name: str = "unknown") -> Gauge:
    """Get or create the server draining gauge metric.

    This metric indicates when a server is in graceful shutdown mode,
    allowing load balancers (like llm-d EPP) to stop routing new requests.
    """
    global _server_draining_gauge
    if _server_draining_gauge is None:
        _server_draining_gauge = Gauge(
            name="vllm:server_draining",
            documentation=(
                "Server draining state. 1 means the server is shutting down "
                "gracefully and should not receive new requests."
            ),
            labelnames=["model_name"],
            multiprocess_mode="livemax",
        )
        # initialize to 0 (not draining)
        _server_draining_gauge.labels(model_name=model_name).set(0)
    return _server_draining_gauge


def set_server_draining(model_name: str = "unknown", draining: bool = True):
    """Set the server draining state.

    Args:
        model_name: The model name label for the metric
        draining: True if server is draining, False otherwise
    """
    gauge = get_server_draining_gauge(model_name)
    gauge.labels(model_name=model_name).set(1 if draining else 0)


# Global temporary directory for prometheus multiprocessing
_prometheus_multiproc_dir: tempfile.TemporaryDirectory | None = None


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
        multiprocess.MultiProcessCollector(registry)
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
        if hasattr(collector, "_name") and "vllm" in collector._name:
            registry.unregister(collector)


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
