# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
import time
from collections import Counter
from contextlib import suppress
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class GCDebugConfig:
    """
    Config for GC Debugger.
    - 0: disable GC debugger
    - 1: enable GC debugger with gc.collect elapsed times
    - '{"top_objects":5}': enable GC debugger with top 5 collected objects
    """

    def __init__(self, gc_debug_conf: str | None = None) -> None:
        self.enabled: bool = False
        self.top_objects: int = -1

        if not gc_debug_conf or gc_debug_conf == "0":
            pass
        elif gc_debug_conf == "1":
            self.enabled = True
        else:
            try:
                json_conf = json.loads(gc_debug_conf)
                self.enabled = True
                self.top_objects = json_conf.get("top_objects", -1)
            except Exception:
                self.enabled = False
                logger.error("Failed to parse VLLM_GC_DEBUG(%s)", envs.VLLM_GC_DEBUG)
        logger.debug("GC Debug Config. %s", str(self))

    def __repr__(self) -> str:
        return f"enabled:{self.enabled},top_objects:{self.top_objects}"


class GCDebugger:
    """
    Debugger for GC which logs helpful information for GC understanding.
    To enable, you should call maybe_attach_gc_debug_callback in the process.
    """

    def __init__(self, config: GCDebugConfig) -> None:
        self.config = config
        # Start time in micro second of this GC cycle
        self.start_time_ns: int = time.monotonic_ns()
        self.num_objects: int = 0
        # If config.top_objects is positive,
        # compute top collected objects by object types
        self.gc_top_collected_objects: str = ""

    def handle(self, phase: str, info: dict[str, int]) -> None:
        """
        Handles a GC event (e.g. GC start or GC finish)
        """
        generation = info.get("generation")
        if generation is None:
            return
        if phase == "start":
            # Before GC started, record GC start time
            # and top collected objects
            self.start_time_ns = time.monotonic_ns()
            objects = gc.get_objects(generation)
            self.num_objects = len(objects)
            self.gc_top_collected_objects = _compute_top_gc_collected_objects(
                objects, self.config.top_objects
            )
        elif phase == "stop":
            # After GC finished, Record GC elapsed time and
            # optionally top collected objects
            elpased_ms = (time.monotonic_ns() - self.start_time_ns) / 1e6
            logger.info(
                "GC took %.3fms to complete. "
                "Collected %s objects (out of %d) in GC generation %d.%s",
                elpased_ms,
                str(info.get("collected", "?")),
                self.num_objects,
                generation,
                (
                    f" Top collected objects: \n{self.gc_top_collected_objects}"
                    if self.gc_top_collected_objects
                    else ""
                ),
            )


def freeze_gc_heap() -> None:
    """
    Freeze all objects tracked by the garbage collector. It should be invoked
    after server init / warmup, to reduce GC overhead from static objects
    during serving time.
    """
    # Ensure all static objects are pushed down to the oldest generation for
    # freeze
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    # Freeze all GC tracked objects
    gc.freeze()


def maybe_attach_gc_debug_callback() -> None:
    """
    Attached a callback for GC debug when VLLM_GC_DEBUG is enabled.
    """
    config = GCDebugConfig(envs.VLLM_GC_DEBUG)
    if config.enabled:
        debugger: GCDebugger = GCDebugger(config)

        def gc_callback(phase: str, info: dict[str, int]) -> None:
            debugger.handle(phase, info)

        gc.callbacks.append(gc_callback)


def _compute_detailed_type(o: Any) -> str:
    """
    Detailed object type.

    TODO(Jialin): Further enhance the detailed type with element types for
    easier debugging. We tried but occasionally it would run into signals
    which kills the engine.
    """
    size_str: str = ""
    # Object doesn't support len() - this can happen with type objects
    # or other objects that don't implement __len__ properly
    with suppress(Exception):
        size_str = f"(size:{len(o)})"
    return f"{str(type(o))}{size_str}"


def _compute_top_gc_collected_objects(objects: list[Any], top: int) -> str:
    """
    Group collected objects by types.
    """
    if top <= 0:
        return ""
    object_types = [_compute_detailed_type(o) for o in objects]
    return "\n".join(
        f"{count:>5}:{object_type}"
        for object_type, count in Counter(object_types).most_common(top)
    )


class ManualGCController:
    """
    Controller for manual garbage collection to avoid GC pauses during
    GPU kernel execution.

    Problem:
        Python's automatic GC can trigger at any time during object allocation,
        causing "stop the world" pauses that delay GPU kernel launches and
        reduce GPU utilization.

    Solution:
        Disable automatic GC and invoke gc.collect() manually at controlled
        points where CPU is already waiting (e.g., during CUDA stream sync).
        This hides GC latency behind GPU compute time.

    Hook Point Rationale:
        The GC is invoked in EngineCore.step() just BEFORE future.result():

            gc_collect_on_sync()      # <-- GC runs here
            model_output = future.result()  # CPU blocks waiting for GPU

        Why this location is ideal:
        1. CPU is about to block anyway - GC latency is "free"
        2. Outside any locks or critical sections
        3. Not in hot loops (once per engine step, not per token)
        4. GPU is executing - GC hidden behind compute time

    Rate Limiting:
        - Minimum interval between GC calls (default: 10ms)
        - Respects Python's allocation thresholds (700, 10, 10)
        - Prevents GC thrashing under rapid allocation patterns

    Safety Features:
        - Off by default (opt-in via VLLM_MANUAL_GC_CONTROL=1)
        - Leak guard: forces collection if allocations exceed 10x threshold
        - Lightweight telemetry: count, total time, max time, per-generation stats
        - Can be disabled at runtime if issues detected

    Topologies Tested:
        - Single GPU (sync scheduling)
        - Single GPU (async scheduling)
        - Multi-GPU tensor parallel
        - Multiprocessing executor (VLLM_ENABLE_V1_MULTIPROCESSING=1)

    Usage:
        controller = ManualGCController.get_instance()
        controller.enable()  # Disables automatic GC

        # At sync points (e.g., before blocking on future.result()):
        controller.maybe_collect()

        # Get telemetry:
        stats = controller.get_stats()
    """

    _instance: "ManualGCController | None" = None

    # Default GC thresholds from Python (gen0, gen1, gen2)
    DEFAULT_THRESHOLDS = (700, 10, 10)

    # Safety limit: force GC if allocations exceed this multiple of threshold
    LEAK_GUARD_MULTIPLIER = 10

    # Minimum interval between GC calls in seconds (rate limiting)
    MIN_GC_INTERVAL_S = 0.010  # 10ms

    def __init__(self) -> None:
        self._enabled = False
        self._original_thresholds: tuple[int, int, int] = gc.get_threshold()

        # Track generation collection counts for threshold logic
        self._gc0_count_since_gc1 = 0
        self._gc1_count_since_gc2 = 0

        # Rate limiting: track last GC time
        self._last_gc_time_ns: int = 0

        # Telemetry - lightweight metrics
        self._total_gc_time_ms = 0.0
        self._gc_invocations = 0
        self._objects_collected = 0
        self._max_gc_time_ms = 0.0
        self._skipped_due_to_rate_limit = 0

        # Per-generation telemetry
        self._gen_invocations = [0, 0, 0]  # gen0, gen1, gen2
        self._gen_time_ms = [0.0, 0.0, 0.0]
        self._gen_objects = [0, 0, 0]

        # Use configured or default thresholds
        self._thresholds = self.DEFAULT_THRESHOLDS

    @classmethod
    def get_instance(cls) -> "ManualGCController":
        """Get singleton instance of ManualGCController."""
        if cls._instance is None:
            cls._instance = ManualGCController()
        return cls._instance

    def enable(self) -> None:
        """
        Enable manual GC control by disabling automatic GC.
        Should be called once during engine initialization.
        """
        if self._enabled:
            return

        # Store original thresholds for potential restore
        self._original_thresholds = gc.get_threshold()

        # Disable automatic GC
        gc.disable()
        self._enabled = True
        self._last_gc_time_ns = time.monotonic_ns()

        logger.info(
            "Manual GC control enabled. Automatic GC disabled. "
            "Thresholds: %s, Leak guard: %dx, Min interval: %.0fms",
            self._thresholds,
            self.LEAK_GUARD_MULTIPLIER,
            self.MIN_GC_INTERVAL_S * 1000,
        )

    def disable(self) -> None:
        """
        Disable manual GC control and restore automatic GC.
        """
        if not self._enabled:
            return

        gc.enable()
        gc.set_threshold(*self._original_thresholds)
        self._enabled = False

        logger.info(
            "Manual GC control disabled. Automatic GC restored. "
            "Final stats: invocations=%d, total_time=%.2fms, "
            "max_time=%.2fms, objects_collected=%d, rate_limited=%d",
            self._gc_invocations,
            self._total_gc_time_ms,
            self._max_gc_time_ms,
            self._objects_collected,
            self._skipped_due_to_rate_limit,
        )

    def maybe_collect(self) -> int:
        """
        Check if GC should be triggered based on thresholds and collect if needed.

        This should be called at controlled points where CPU is waiting,
        such as just before blocking on future.result() during CUDA sync.

        Rate limiting prevents GC thrashing - calls within MIN_GC_INTERVAL_S
        of the last GC are skipped (unless leak guard triggers).

        Returns:
            Number of objects collected (0 if no collection performed).
        """
        if not self._enabled:
            return 0

        current_time_ns = time.monotonic_ns()

        # Get current GC counts to estimate allocations
        counts = gc.get_count()
        gen0_count = counts[0]

        # Check leak guard first (bypasses rate limit)
        if gen0_count >= self._thresholds[0] * self.LEAK_GUARD_MULTIPLIER:
            logger.warning(
                "Leak guard triggered: gen0_count=%d exceeds safety limit. "
                "Forcing full GC collection.",
                gen0_count,
            )
            collected = self._do_collect(2)
            self._gc0_count_since_gc1 = 0
            self._gc1_count_since_gc2 = 0
            return collected

        # Rate limiting check
        elapsed_ns = current_time_ns - self._last_gc_time_ns
        if elapsed_ns < self.MIN_GC_INTERVAL_S * 1e9:
            self._skipped_due_to_rate_limit += 1
            return 0

        # Check if we need to collect based on thresholds
        if gen0_count < self._thresholds[0]:
            return 0

        # Determine the highest generation to collect to avoid redundant work.
        # Note: gc.collect(gen) collects gen and all younger generations.
        pending_gc0_count = self._gc0_count_since_gc1 + 1
        highest_gen = 0
        if pending_gc0_count >= self._thresholds[1]:
            highest_gen = 1
            pending_gc1_count = self._gc1_count_since_gc2 + 1
            if pending_gc1_count >= self._thresholds[2]:
                highest_gen = 2

        total_collected = self._do_collect(highest_gen)

        if highest_gen == 0:
            self._gc0_count_since_gc1 += 1
        elif highest_gen == 1:
            self._gc0_count_since_gc1 = 0
            self._gc1_count_since_gc2 += 1
        else:
            self._gc0_count_since_gc1 = 0
            self._gc1_count_since_gc2 = 0

        return total_collected

    def force_collect(self, generation: int = 2) -> int:
        """
        Force a GC collection regardless of thresholds and rate limits.

        Args:
            generation: GC generation to collect (0, 1, or 2).

        Returns:
            Number of objects collected.
        """
        if not self._enabled:
            # Even if not enabled, allow forced collection
            return gc.collect(generation)

        return self._do_collect(generation)

    def _do_collect(self, generation: int) -> int:
        """Perform GC collection with timing and telemetry."""
        start_time_ns = time.monotonic_ns()
        collected = gc.collect(generation)
        elapsed_ms = (time.monotonic_ns() - start_time_ns) / 1e6

        # Update global telemetry
        self._total_gc_time_ms += elapsed_ms
        self._gc_invocations += 1
        self._objects_collected += collected
        self._max_gc_time_ms = max(self._max_gc_time_ms, elapsed_ms)
        self._last_gc_time_ns = time.monotonic_ns()

        # Update per-generation telemetry
        self._gen_invocations[generation] += 1
        self._gen_time_ms[generation] += elapsed_ms
        self._gen_objects[generation] += collected

        if envs.VLLM_GC_DEBUG:
            logger.debug(
                "Manual GC gen%d: collected %d objects in %.3fms "
                "(total: %d invocations, %.2fms)",
                generation,
                collected,
                elapsed_ms,
                self._gc_invocations,
                self._total_gc_time_ms,
            )

        return collected

    def get_stats(self) -> dict[str, Any]:
        """
        Get GC telemetry for monitoring and debugging.

        Returns lightweight metrics suitable for periodic logging
        or metrics export.
        """
        return {
            "enabled": self._enabled,
            # Global stats
            "gc_invocations": self._gc_invocations,
            "total_gc_time_ms": round(self._total_gc_time_ms, 2),
            "max_gc_time_ms": round(self._max_gc_time_ms, 2),
            "avg_gc_time_ms": round(
                self._total_gc_time_ms / self._gc_invocations
                if self._gc_invocations > 0
                else 0.0,
                2,
            ),
            "objects_collected": self._objects_collected,
            "skipped_rate_limited": self._skipped_due_to_rate_limit,
            # Per-generation stats
            "gen0_invocations": self._gen_invocations[0],
            "gen1_invocations": self._gen_invocations[1],
            "gen2_invocations": self._gen_invocations[2],
            "gen0_time_ms": round(self._gen_time_ms[0], 2),
            "gen1_time_ms": round(self._gen_time_ms[1], 2),
            "gen2_time_ms": round(self._gen_time_ms[2], 2),
        }

    def log_stats(self) -> None:
        """Log current GC telemetry at INFO level."""
        if not self._enabled:
            return
        stats = self.get_stats()
        logger.info(
            "Manual GC stats: invocations=%d, total=%.2fms, max=%.2fms, "
            "avg=%.2fms, collected=%d, rate_limited=%d | "
            "gen0=%d/%.1fms, gen1=%d/%.1fms, gen2=%d/%.1fms",
            stats["gc_invocations"],
            stats["total_gc_time_ms"],
            stats["max_gc_time_ms"],
            stats["avg_gc_time_ms"],
            stats["objects_collected"],
            stats["skipped_rate_limited"],
            stats["gen0_invocations"],
            stats["gen0_time_ms"],
            stats["gen1_invocations"],
            stats["gen1_time_ms"],
            stats["gen2_invocations"],
            stats["gen2_time_ms"],
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def maybe_enable_manual_gc_control() -> ManualGCController | None:
    """
    Enable manual GC control if VLLM_MANUAL_GC_CONTROL is set.

    Returns:
        ManualGCController instance if enabled, None otherwise.
    """
    if envs.VLLM_MANUAL_GC_CONTROL:
        controller = ManualGCController.get_instance()
        controller.enable()
        return controller
    return None


def gc_collect_on_sync() -> int:
    """
    Invoke manual GC at CUDA synchronization points.

    This is the primary hook for manual GC control. It should be called
    at points where CPU is about to block waiting for GPU, hiding GC
    latency behind GPU compute time.

    Hook Point Rationale:
        Called in EngineCore.step() BEFORE future.result():

            gc_collect_on_sync()           # <-- Here: CPU about to wait
            model_output = future.result()  # CPU blocks for GPU

        Why this location:
        1. CPU is about to block anyway - GC latency is effectively "free"
        2. Outside any locks - no risk of deadlock or contention
        3. Not in hot loops - once per engine step, not per token
        4. GPU is actively computing - GC hidden behind GPU work

    Safety:
        - No-op if manual GC control is not enabled
        - Rate-limited to prevent GC thrashing
        - Respects allocation thresholds

    Returns:
        Number of objects collected (0 if GC not enabled, rate-limited,
        or thresholds not met).
    """
    controller = ManualGCController.get_instance()
    if controller.is_enabled:
        return controller.maybe_collect()
    return 0
