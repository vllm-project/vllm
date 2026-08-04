# SPDX-License-Identifier: Apache-2.0

"""Trace recorder lifecycle helpers.

Used by the cache server entry points (``server.py`` and
``http_server.py``) to construct, register, and tear down trace
recorders alongside the EventBus.
"""

# Future
from __future__ import annotations

# Standard
from datetime import datetime, timezone
import os
import tempfile

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.mp_observability.event_bus import EventBus
from lmcache.v1.mp_observability.trace.recorder import (
    StorageTraceRecorder,
    TraceRecorder,
)

logger = init_logger(__name__)


def _default_trace_path() -> str:
    """Mint a timestamped path for an unnamed trace file."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        tempfile.gettempdir(),
        f"lmcache-trace-{os.getpid()}-{stamp}.lct",
    )


def maybe_initialize_trace_recorder(
    bus: EventBus,
    obs_config: ObservabilityConfig,
    storage_manager_config: StorageManagerConfig,
) -> TraceRecorder | None:
    """Construct and register a trace recorder if configured.

    Args:
        bus: The active EventBus to subscribe the recorder to.
        obs_config: Observability config carrying the trace flags.
        storage_manager_config: The StorageManagerConfig in use.  Used
            to populate the trace file's header digest so a replay
            driver can detect mismatched configurations.

    Returns:
        The created recorder, or ``None`` when ``obs_config.trace_level``
        is unset.

    The recorder is registered on the bus, so :meth:`EventBus.stop`
    will invoke its ``shutdown`` (which flushes and closes the file).
    Callers do not need to track the returned reference for cleanup;
    it is returned only for testing and observation.
    """
    level = obs_config.trace_level
    if not level:
        return None
    if level != "storage":
        raise ValueError(
            f"unsupported trace level {level!r}; only 'storage' is supported"
        )

    output_path = obs_config.trace_output or _default_trace_path()
    if obs_config.trace_output is None:
        logger.info(
            "trace recording enabled (level=%s); no --trace-output given, "
            "writing to %s",
            level,
            output_path,
        )

    recorder = StorageTraceRecorder(output_path=output_path)
    recorder.attach_storage_config(storage_manager_config)
    bus.register_subscriber(recorder)
    return recorder
