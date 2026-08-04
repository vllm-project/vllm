# SPDX-License-Identifier: Apache-2.0
"""
Unified PeriodicThread abstraction for LMCache background threads.

This module provides a standardized way to create and manage periodic
background threads with proper naming, monitoring, and lifecycle management.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.exceptions import IrrecoverableException

if TYPE_CHECKING:
    # First Party
    pass

logger = init_logger(__name__)


class ThreadLevel(Enum):
    """
    Thread importance level.

    CRITICAL: Thread failure causes severe system degradation or data loss.
              Examples: health-monitor-thread, PinMonitor-thread

    HIGH: Thread failure significantly impacts performance or functionality.
          Examples: storage-manager-event-loop, lookup-server threads

    MEDIUM: Thread failure causes noticeable degradation but system remains functional.
            Examples: stats-logger-thread, batched-message-sender-thread

    LOW: Thread failure has minimal impact on system operation.
         Examples: lazy-memory-expand-thread
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ThreadRunSummary:
    """Summary of a single thread execution cycle."""

    timestamp: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    message: str = ""
    extra_info: Dict[str, str] = field(default_factory=dict)


class PeriodicThread(ABC):
    """
    Abstract base class for periodic background threads.

    This class provides a standardized framework for creating periodic
    background threads with proper naming, monitoring, and lifecycle management.

    Attributes:
        name: Human-readable name of the thread
        interval: Time interval between executions in seconds
        level: Importance level of the thread
        init_wait: Initial wait time before first execution in seconds

    Features:
        - Named threads for easy identification
        - Configurable execution interval
        - Initial wait time before first execution
        - Automatic tracking of last run time and summary
        - Thread level classification for monitoring
        - Interruptible sleep for graceful shutdown

    Usage:
        class MyPeriodicThread(PeriodicThread):
            def __init__(self):
                super().__init__(
                    name="my-periodic-thread",
                    interval=30.0,
                    level=ThreadLevel.MEDIUM,
                    init_wait=5.0,
                )

            def _execute(self) -> ThreadRunSummary:
                # Perform periodic work
                return ThreadRunSummary(
                    timestamp=time.time(),
                    success=True,
                    message="Completed successfully"
                )

        thread = MyPeriodicThread()
        thread.start()
        # ... later ...
        thread.stop()
    """

    def __init__(
        self,
        name: str,
        interval: float,
        level: ThreadLevel = ThreadLevel.MEDIUM,
        init_wait: float = 0.0,
    ):
        """
        Initialize a PeriodicThread.

        Args:
            name: Thread name for identification
            interval: Execution interval in seconds
            level: Thread importance level
            init_wait: Initial wait time before first execution in seconds
        """
        self._name = name
        self._interval = interval
        self._level = level
        self._init_wait = init_wait

        # Thread state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Set to break out of the interval sleep early: either because
        # stop() was called or because wake() requested an immediate run.
        self._wake_event = threading.Event()
        self._running = False

        # Execution tracking
        self._last_run_time: float = 0.0
        self._last_summary: Optional[ThreadRunSummary] = None
        self._total_runs: int = 0
        self._failed_runs: int = 0
        self._lock = threading.RLock()

        # Start time (set when thread starts)
        self._start_time: float = 0.0

    @property
    def name(self) -> str:
        """Get the thread name."""
        return self._name

    @property
    def interval(self) -> float:
        """Get the execution interval in seconds."""
        return self._interval

    @property
    def level(self) -> ThreadLevel:
        """Get the thread importance level."""
        return self._level

    @property
    def init_wait(self) -> float:
        """Get the initial wait time in seconds."""
        return self._init_wait

    @property
    def last_run_time(self) -> float:
        """Get the timestamp of the last execution."""
        with self._lock:
            return self._last_run_time

    @property
    def last_summary(self) -> Optional[ThreadRunSummary]:
        """Get the summary of the last execution."""
        with self._lock:
            return self._last_summary

    @property
    def total_runs(self) -> int:
        """Get the total number of executions."""
        with self._lock:
            return self._total_runs

    @property
    def failed_runs(self) -> int:
        """Get the number of failed executions."""
        with self._lock:
            return self._failed_runs

    @property
    def is_running(self) -> bool:
        """Check if the thread is currently running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def stop_requested(self) -> bool:
        """Whether a stop has been requested (set by stop, reset by start)."""
        return self._stop_event.is_set()

    @property
    def is_active(self) -> bool:
        """
        Check if the thread is active (running and recently executed).

        A thread is considered active if:
        1. It is running
        2. The time since last run is less than 3 * interval

        Returns:
            bool: True if active, False otherwise
        """
        if not self.is_running:
            return False

        with self._lock:
            if self._last_run_time == 0:
                # Thread started but hasn't run yet
                # Consider active if within init_wait + 3 * interval from start
                time_since_start = time.time() - self._start_time
                return time_since_start < self._init_wait + 3 * self._interval

            time_since_last_run = time.time() - self._last_run_time
            return time_since_last_run < 3 * self._interval

    def start(self) -> Optional[threading.Thread]:
        """
        Start the periodic thread.

        Returns:
            Optional[threading.Thread]: The started thread, or None if already running
        """
        if self._running:
            logger.warning("PeriodicThread %s is already running", self._name)
            return None

        self._stop_event.clear()
        self._wake_event.clear()
        self._running = True
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=self._name,
        )
        self._thread.start()

        logger.info(
            "Started PeriodicThread: %s (level=%s, interval=%.1fs, init_wait=%.1fs)",
            self._name,
            self._level.value,
            self._interval,
            self._init_wait,
        )
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the periodic thread.

        Args:
            timeout: Maximum time to wait for thread termination in seconds
        """
        if not self._running:
            return

        logger.info("Stopping PeriodicThread: %s", self._name)
        self._running = False
        self._stop_event.set()
        # Break the interval sleep so the loop notices stop immediately.
        self._wake_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "PeriodicThread %s did not terminate within %.1fs timeout",
                    self._name,
                    timeout,
                )

    def wake(self) -> None:
        """Trigger one execution now instead of waiting for the next tick.

        Safe to call from any thread. No-op if the thread is not running.
        If wake() is called during an execution, the next interval sleep
        returns immediately and another cycle runs right after.
        """
        if not self._running:
            return
        self._wake_event.set()

    def _run_loop(self) -> None:
        """Main thread loop."""
        # Initial wait
        if self._init_wait > 0:
            if self._stop_event.wait(timeout=self._init_wait):
                logger.info("PeriodicThread %s stopped during init_wait", self._name)
                return
            # A wake() during init_wait should not count as stop; only
            # honor stop_event above.
            self._wake_event.clear()

        logger.info(
            "PeriodicThread %s entering main loop (interval=%.1fs)",
            self._name,
            self._interval,
        )

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                summary = self._execute()
                summary.timestamp = start_time
                summary.duration_ms = (time.time() - start_time) * 1000

                with self._lock:
                    self._last_run_time = start_time
                    self._last_summary = summary
                    self._total_runs += 1
                    if not summary.success:
                        self._failed_runs += 1

            except IrrecoverableException as e:
                logger.error(
                    "IrrecoverableException in PeriodicThread %s: %s",
                    self._name,
                    e,
                    exc_info=True,
                )
                summary = ThreadRunSummary(
                    timestamp=start_time,
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    message=str(e),
                )
                with self._lock:
                    self._last_run_time = start_time
                    self._last_summary = summary
                    self._total_runs += 1
                    self._failed_runs += 1
                # Stop the loop on irrecoverable exceptions
                logger.info(
                    "PeriodicThread %s stopping due to IrrecoverableException",
                    self._name,
                )
                break
            except Exception as e:
                logger.error(
                    "Error in PeriodicThread %s: %s", self._name, e, exc_info=True
                )
                summary = ThreadRunSummary(
                    timestamp=start_time,
                    duration_ms=(time.time() - start_time) * 1000,
                    success=False,
                    message=str(e),
                )
                with self._lock:
                    self._last_run_time = start_time
                    self._last_summary = summary
                    self._total_runs += 1
                    self._failed_runs += 1

            # Wait for next interval; wake() can shorten this wait.
            self._wake_event.wait(timeout=self._interval)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break

        logger.info("PeriodicThread %s loop stopped", self._name)

    @abstractmethod
    def _execute(self) -> ThreadRunSummary:
        """
        Execute one cycle of the periodic task.

        This method should be overridden by subclasses to implement
        the actual periodic work.

        Returns:
            ThreadRunSummary: Summary of this execution cycle
        """
        pass

    def get_status(self) -> Dict:
        """
        Get the current status of the thread.

        Returns:
            Dict: Thread status information
        """
        with self._lock:
            last_summary_dict = None
            if self._last_summary:
                last_summary_dict = {
                    "timestamp": self._last_summary.timestamp,
                    "duration_ms": self._last_summary.duration_ms,
                    "success": self._last_summary.success,
                    "message": self._last_summary.message,
                    "extra_info": self._last_summary.extra_info,
                }

            return {
                "name": self._name,
                "level": self._level.value,
                "interval": self._interval,
                "init_wait": self._init_wait,
                "is_running": self.is_running,
                "is_active": self.is_active,
                "last_run_time": self._last_run_time,
                "last_run_ago": time.time() - self._last_run_time
                if self._last_run_time > 0
                else None,
                "total_runs": self._total_runs,
                "failed_runs": self._failed_runs,
                "success_rate": (
                    (self._total_runs - self._failed_runs) / self._total_runs * 100
                    if self._total_runs > 0
                    else None
                ),
                "last_summary": last_summary_dict,
            }


class PeriodicThreadRegistry:
    """
    Global registry for all PeriodicThread instances.

    This class provides a centralized way to track and manage all
    periodic threads in the system.
    """

    _instance: Optional["PeriodicThreadRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._threads: Dict[str, PeriodicThread] = {}
        self._registry_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "PeriodicThreadRegistry":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PeriodicThreadRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Mainly for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unregister_all()
            cls._instance = None

    def register(self, thread: PeriodicThread) -> None:
        """
        Register a periodic thread.

        Args:
            thread: The PeriodicThread to register
        """
        with self._registry_lock:
            if thread.name in self._threads:
                logger.warning(
                    "PeriodicThread %s is already registered, replacing",
                    thread.name,
                )
            self._threads[thread.name] = thread
            logger.debug("Registered PeriodicThread: %s", thread.name)

    def unregister(self, name: str) -> Optional[PeriodicThread]:
        """
        Unregister a periodic thread by name.

        Args:
            name: The name of the thread to unregister

        Returns:
            The unregistered thread, or None if not found
        """
        with self._registry_lock:
            thread = self._threads.pop(name, None)
            if thread:
                logger.debug("Unregistered PeriodicThread: %s", name)
            return thread

    def unregister_all(self) -> None:
        """Unregister all periodic threads."""
        with self._registry_lock:
            self._threads.clear()
            logger.debug("Unregistered all PeriodicThreads")

    def get(self, name: str) -> Optional[PeriodicThread]:
        """Get a registered thread by name."""
        with self._registry_lock:
            return self._threads.get(name)

    def get_all(self) -> List[PeriodicThread]:
        """Get all registered threads."""
        with self._registry_lock:
            return list(self._threads.values())

    def get_by_level(self, level: ThreadLevel) -> List[PeriodicThread]:
        """Get all threads with the specified level."""
        with self._registry_lock:
            return [t for t in self._threads.values() if t.level == level]

    def get_running_count(self) -> int:
        """Get the count of running threads."""
        with self._registry_lock:
            return sum(1 for t in self._threads.values() if t.is_running)

    def get_active_count(self) -> int:
        """Get the count of active threads."""
        with self._registry_lock:
            return sum(1 for t in self._threads.values() if t.is_active)

    def get_count_by_level(self, level: ThreadLevel) -> Dict[str, int]:
        """
        Get counts of threads for a specific level.

        Returns:
            Dict with keys: total, running, active
        """
        with self._registry_lock:
            threads = [t for t in self._threads.values() if t.level == level]
            return {
                "total": len(threads),
                "running": sum(1 for t in threads if t.is_running),
                "active": sum(1 for t in threads if t.is_active),
            }

    def get_summary(self) -> Dict:
        """
        Get a summary of all registered threads.

        Returns:
            Dict containing:
                - total_count: Total number of registered threads
                - running_count: Number of running threads
                - active_count: Number of active threads
                - by_level: Counts by thread level
                - threads: List of thread statuses
        """
        with self._registry_lock:
            by_level = {}
            for level in ThreadLevel:
                by_level[level.value] = self.get_count_by_level(level)

            return {
                "total_count": len(self._threads),
                "running_count": self.get_running_count(),
                "active_count": self.get_active_count(),
                "by_level": by_level,
                "threads": [t.get_status() for t in self._threads.values()],
            }


def create_periodic_thread(
    name: str,
    interval: float,
    execute_fn: Callable[[], ThreadRunSummary],
    level: ThreadLevel = ThreadLevel.MEDIUM,
    init_wait: float = 0.0,
    auto_register: bool = True,
) -> PeriodicThread:
    """
    Factory function to create a simple PeriodicThread.

    This is a convenience function for creating periodic threads
    without needing to define a subclass.

    Args:
        name: Thread name
        interval: Execution interval in seconds
        execute_fn: Function to execute on each cycle
        level: Thread importance level
        init_wait: Initial wait time before first execution
        auto_register: Whether to automatically register with the global registry

    Returns:
        PeriodicThread: The created thread instance
    """

    class SimplePeriodicThread(PeriodicThread):
        def __init__(self):
            super().__init__(
                name=name,
                interval=interval,
                level=level,
                init_wait=init_wait,
            )
            self._execute_fn = execute_fn

        def _execute(self) -> ThreadRunSummary:
            return self._execute_fn()

    thread = SimplePeriodicThread()

    if auto_register:
        PeriodicThreadRegistry.get_instance().register(thread)

    return thread
