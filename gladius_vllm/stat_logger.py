"""Optional, secondary GLADIUS stat-logger plugin.

`StatLoggerBase.record()` receives only (scheduler_stats, iteration_stats,
mm_cache_stats, engine_idx) -- no reference to the Scheduler instance or the
per-step SchedulerOutput -- so this class cannot reproduce the full
telemetry.jsonl feed on its own; that's `GladiusScheduler`/`TelemetryWriter`'s
job (they have direct, always-in-process access to everything they need).

This class exists only for users who also want the SchedulerStats-only
subset of these numbers surfaced through vLLM's standard logging/Prometheus
stat-logger plumbing (`vllm.stat_logger_plugins` entry-point group, or
`AsyncLLM(stat_loggers=[...])`). It is NOT load-bearing for the
GladiusScheduler <-> control-plane protocol.

Known limitation (see design plan "Open risks"): in default
VLLM_ENABLE_V1_MULTIPROCESSING=1 mode it is unconfirmed whether this class
runs in the same OS process as the Scheduler it looks up via
gladius_vllm.registry; if not, get_scheduler() below simply returns None and
this logger degrades to reporting nothing extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.metrics.loggers import StatLoggerBase

from gladius_vllm.registry import get_scheduler
from gladius_vllm.schema import resolve_engine_id

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats


class GladiusStatLogger(StatLoggerBase):
    def __init__(self, vllm_config: "VllmConfig", engine_index: int = 0) -> None:
        self._engine_id = resolve_engine_id(vllm_config)
        self._engine_index = engine_index

    def record(
        self,
        scheduler_stats: "SchedulerStats | None",
        iteration_stats: "IterationStats | None",
        mm_cache_stats: "MultiModalCacheStats | None" = None,
        engine_idx: int = 0,
    ) -> None:
        # Lazy per-call lookup (not cached at __init__) since construction
        # order between GladiusScheduler and stat loggers isn't guaranteed.
        scheduler = get_scheduler(self._engine_id)
        if scheduler is None or scheduler_stats is None:
            return
        # Intentionally a no-op beyond the lookup for phase 1: GladiusScheduler
        # already writes the authoritative telemetry.jsonl line itself. This
        # hook is reserved for future Prometheus/console-log wiring.

    def log_engine_initialized(self) -> None:
        pass
