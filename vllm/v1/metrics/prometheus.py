# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile
from typing import Optional

from prometheus_client import (REGISTRY, CollectorRegistry, make_asgi_app,
                               multiprocess)
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.routing import Mount

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global temporary directory for prometheus multiprocessing
_prometheus_multiproc_dir: Optional[tempfile.TemporaryDirectory] = None


def setup_multiprocess_prometheus():
    """Set up prometheus multiprocessing directory if not already configured.
    
    """
    global _prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        # Make TemporaryDirectory for prometheus multiprocessing
        # Note: global TemporaryDirectory will be automatically
        # cleaned up upon exit.
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
        logger.debug("Created PROMETHEUS_MULTIPROC_DIR at %s",
                     _prometheus_multiproc_dir.name)
    else:
        logger.warning("Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                       "This directory must be wiped between vLLM runs or "
                       "you will find inaccurate metrics. Unset the variable "
                       "and vLLM will properly handle cleanup.")


def unregister_vllm_metrics():
    """Unregister any existing vLLM collectors from the prometheus registry.
    
    This is useful for testing and CI/CD where metrics may be registered
    multiple times across test runs.
    """
    # Unregister any existing vLLM collectors
    for collector in list(REGISTRY._collector_to_names):
        if hasattr(collector, "_name") and "vllm" in collector._name:
            REGISTRY.unregister(collector)


def get_prometheus_registry():
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


def mount_metrics(app):
    """Mount prometheus metrics to a FastAPI app.
    
    Args:
        app: FastAPI application
    """

    registry = get_prometheus_registry()

    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def mark_process_dead(pid: int):
    """Mark a process as dead in prometheus multiprocessing.
    
    Args:
        pid: Process ID to mark as dead
    """
    multiprocess.mark_process_dead(pid)
