# SPDX-License-Identifier: Apache-2.0

"""CPython garbage-collector timing for the MP server process.

A full (generation-2) collection is stop-the-world and walks the whole heap,
so it lands inside store/retrieve handling as tail latency that nothing else
in the server accounts for.  :class:`GCMonitor` times every collection via a
``gc.callbacks`` hook and logs the slow ones.

The monitor logs directly rather than publishing to the EventBus: the hook
runs inside the collector, on whichever thread triggered it, so it does the
least work it can and GC timing stays decoupled from the bus's queue depth
and drain lag.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from dataclasses import dataclass
import gc
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class GCMonitorConfig:
    """Configuration for :class:`GCMonitor`.

    Attributes:
        enabled: When ``False`` no hook is installed and the process pays
            nothing.
        min_pause_ms: Collections faster than this are not logged.  ``0.0``
            logs everything, including the sub-millisecond gen-0 sweeps
            CPython runs roughly every 700 net container allocations.
        top_objects: When positive, log a breakdown of the ``N`` most common
            object types in the generation being collected.  Walks the whole
            generation on *every* collection (O(heap)) -- debugging only.
    """

    enabled: bool = False
    min_pause_ms: float = 1.0
    top_objects: int = 0

    def __post_init__(self) -> None:
        if self.min_pause_ms < 0.0:
            raise ValueError(f"min_pause_ms must be >= 0; got {self.min_pause_ms}")
        if self.top_objects < 0:
            raise ValueError(f"top_objects must be >= 0; got {self.top_objects}")


class GCMonitor:
    """Times CPython garbage collections and logs the slow ones.

    State is mutated only inside the ``gc.callbacks`` hook.  CPython runs at
    most one collection at a time and never re-enters the collector from a
    callback, so no locking is needed even though collections fire on
    arbitrary threads.
    """

    def __init__(self, config: GCMonitorConfig) -> None:
        self._config = config
        self._installed = False
        self._start_ns = 0
        self._top_objects = ""

    def install(self) -> None:
        """Start timing collections.  Idempotent."""
        if self._installed:
            return
        gc.callbacks.append(self._on_gc)
        self._installed = True
        logger.info(
            "GC monitor on (min_pause=%.1fms, top_objects=%d)",
            self._config.min_pause_ms,
            self._config.top_objects,
        )

    def uninstall(self) -> None:
        """Stop timing collections.  Idempotent."""
        if not self._installed:
            return
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        self._installed = False
        logger.info("GC monitor off")

    @property
    def installed(self) -> bool:
        """Whether the ``gc.callbacks`` hook is currently registered."""
        return self._installed

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        """Handle one ``gc.callbacks`` invocation.

        CPython calls this with *phase* ``"start"`` before a collection and
        ``"stop"`` after it, passing the generation plus, on stop, the
        ``collected`` and ``uncollectable`` counts.
        """
        generation = info.get("generation")
        if generation is None:
            return

        if phase == "start":
            # Scan before the timestamp so this monitor's own cost is never
            # attributed to the collector.
            self._top_objects = self._compute_top_objects(generation)
            self._start_ns = time.monotonic_ns()
            return
        if phase != "stop":
            return

        duration_ms = (time.monotonic_ns() - self._start_ns) / 1e6
        top_objects = self._top_objects
        self._top_objects = ""
        if duration_ms < self._config.min_pause_ms:
            return

        logger.info(
            "GC gen%d %.1fms collected=%d uncollectable=%d%s",
            generation,
            duration_ms,
            info.get("collected", 0),
            info.get("uncollectable", 0),
            f" {top_objects}" if top_objects else "",
        )

    def _compute_top_objects(self, generation: int) -> str:
        """Return a one-line ``type=count`` breakdown, or ``""`` when off."""
        top = self._config.top_objects
        if top <= 0:
            return ""
        counts = Counter(_type_name(o) for o in gc.get_objects(generation))
        return " ".join(
            f"{type_name}={count}" for type_name, count in counts.most_common(top)
        )


def _type_name(obj: object) -> str:
    """Return a printable type name for *obj*."""
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


_global_monitor: GCMonitor | None = None


def get_gc_monitor() -> GCMonitor | None:
    """Return the installed process-wide monitor, or ``None``."""
    return _global_monitor


def init_gc_monitor(config: GCMonitorConfig) -> GCMonitor | None:
    """Install the process-wide GC monitor if *config* enables it.

    Uninstalls any previously installed monitor first, so calling this twice
    never leaves a stale ``gc.callbacks`` entry behind.

    Args:
        config: Monitor configuration.

    Returns:
        The installed monitor, or ``None`` when ``config.enabled`` is False.
    """
    global _global_monitor

    shutdown_gc_monitor()
    if not config.enabled:
        return None

    _global_monitor = GCMonitor(config)
    _global_monitor.install()
    return _global_monitor


def shutdown_gc_monitor() -> None:
    """Uninstall the process-wide monitor.  Safe when none is installed."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.uninstall()
        _global_monitor = None
