# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.metadata import LMCacheMetadata


logger = init_logger(__name__)


class PinMonitor(PeriodicThread):
    """
    Global monitor (singleton per process, shared across all cache engines)
    for pinned TensorMemoryObj instances to handle timeout detection.
    This class runs a background thread that periodically checks for pinned objects
    that have exceeded their timeout duration.
    """

    _instance = None
    _instance_lock = threading.Lock()  # Class-level lock for singleton pattern

    def __init__(self, config: "LMCacheEngineConfig") -> None:
        # Initialize PeriodicThread base class
        super().__init__(
            name="PinMonitor-thread",
            interval=config.pin_check_interval_sec,
            level=ThreadLevel.CRITICAL,
            init_wait=0.0,
        )

        # obj_id is the virtual memory address given by Python's id() function
        self._pinned_objects: dict[
            int, tuple["MemoryObj", float]
        ] = {}  # {obj_id: (memory_obj, register_time)}
        self._objects_lock = threading.Lock()
        self._check_interval = config.pin_check_interval_sec
        self._pin_timeout_sec = config.pin_timeout_sec

        # Register with the global registry
        PeriodicThreadRegistry.get_instance().register(self)

        # Auto-start the monitor on first instance creation
        self.start_monitoring()

    @staticmethod
    def GetOrCreate(
        config: Optional["LMCacheEngineConfig"] = None,
        metadata: Optional["LMCacheMetadata"] = None,
    ) -> "PinMonitor":
        """Get or create the singleton instance.

        Args:
            config: Required for first-time initialization.
                Optional for subsequent calls.
            metadata: Metadata for the label view that should publish
                PinMonitor metrics.

        Raises:
            ValueError: If config is None when creating the instance
                for the first time.
        """
        if PinMonitor._instance is None:
            with PinMonitor._instance_lock:
                if PinMonitor._instance is None:
                    assert config is not None, "config is required"
                    PinMonitor._instance = PinMonitor(config)
        instance = PinMonitor._instance
        assert instance is not None
        if metadata is not None:
            assert config is not None, "config is required to set up metrics"
            instance._setup_metrics(config, metadata)
        return instance

    def on_pin(self, memory_obj: "MemoryObj"):
        """Register a pinned memory object for timeout monitoring.

        Note: The same memory_obj can be pinned multiple times, so this
        function may be called multiple times with the same object.
        Each call updates the register time, effectively resetting the
        timeout countdown.
        """
        obj_id = id(memory_obj)
        with self._objects_lock:
            current_time = time.time()
            self._pinned_objects[obj_id] = (memory_obj, current_time)
            logger.debug(
                "Registered pinned object %s for timeout monitoring at time %.2f",
                obj_id,
                current_time,
            )

    def on_unpin(self, memory_obj: "MemoryObj"):
        """Unregister a memory object from timeout monitoring."""
        obj_id = id(memory_obj)
        with self._objects_lock:
            if obj_id in self._pinned_objects:
                del self._pinned_objects[obj_id]
                logger.debug(
                    "Unregistered pinned object %s from timeout monitoring",
                    obj_id,
                )

    def _check_timeouts(self) -> tuple[int, int, int]:
        """Check all registered pinned objects for timeout.

        Returns:
            tuple: (pinned_count, timeout_count, force_unpin_success_count)
        """
        current_time = time.time()
        timeout_objects = []

        with self._objects_lock:
            pinned_count = len(self._pinned_objects)
            for obj_id, (memory_obj, register_time) in list(
                self._pinned_objects.items()
            ):
                # Check if object is still pinned and has exceeded timeout
                if memory_obj.meta.pin_count > 0:
                    elapsed_time = current_time - register_time
                    if elapsed_time > self._pin_timeout_sec:
                        timeout_objects.append((memory_obj, elapsed_time))

        # Force unpin timeout objects outside the lock to avoid deadlocks
        force_unpin_success_count = 0
        for memory_obj, elapsed_time in timeout_objects:
            try:
                self._force_unpin_timeout_object(memory_obj, elapsed_time)
                force_unpin_success_count += 1
            except Exception as e:
                logger.error(
                    "Error forcing unpin for timeout object %s: %s", id(memory_obj), e
                )
        if force_unpin_success_count > 0:
            logger.warning(
                "Force unpinned %d timeout objects in %d pinned_objects "
                "within %d seconds",
                force_unpin_success_count,
                pinned_count,
                self._pin_timeout_sec,
            )
        else:
            logger.debug(
                "PinMonitor check: pinned_objects=%d, timeout_objects=%d, "
                "force_unpin_success=%d",
                pinned_count,
                len(timeout_objects),
                force_unpin_success_count,
            )

        return pinned_count, len(timeout_objects), force_unpin_success_count

    def _force_unpin_timeout_object(self, memory_obj: "MemoryObj", elapsed_time: float):
        """Force unpin a timeout object and log the event."""
        # Get current pin_count without holding the lock for unpin calls
        # Use nullcontext if memory_obj doesn't have a lock attribute
        obj_lock = getattr(memory_obj, "lock", None) or nullcontext()
        with obj_lock:
            current_pin_count = memory_obj.meta.pin_count
            if current_pin_count <= 0:
                return

            logger.warning(
                "Pin timeout detected for MemoryObj %s. "
                "Pin count: %s, Elapsed time: %.2fs. Forcing unpin to 0.",
                memory_obj.meta.address,
                current_pin_count,
                elapsed_time,
            )

        # Update forced unpin statistics
        LMCStatsMonitor.GetOrCreate().update_forced_unpin_count(1)

        # Call unpin() while pin_count > 0 to properly release resources
        while memory_obj.meta.pin_count > 0:
            memory_obj.unpin()

    def _execute(self) -> ThreadRunSummary:
        """
        Execute one pin monitor check cycle.

        This method is called by the PeriodicThread base class.

        Returns:
            ThreadRunSummary: Summary of the check cycle
        """
        pinned_count, timeout_count, force_unpin_count = self._check_timeouts()

        return ThreadRunSummary(
            success=True,
            message=f"Checked {pinned_count} objects, {timeout_count} timeouts, "
            f"{force_unpin_count} force unpinned",
            extra_info={
                "pinned_count": str(pinned_count),
                "timeout_count": str(timeout_count),
                "force_unpin_count": str(force_unpin_count),
            },
        )

    def start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        if self.is_running:
            return

        # Use base class start method
        self.start()
        logger.info("PinMonitor started")

    def _setup_metrics(
        self,
        config: "LMCacheEngineConfig",
        metadata: "LMCacheMetadata",
    ) -> None:
        prometheus_logger = PrometheusLogger.GetOrCreate(metadata, config=config)
        prometheus_logger.pin_monitor_pinned_objects_count.set_function(
            lambda: len(self._pinned_objects)
        )

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        if not self.is_running:
            return

        # Unregister from the global registry
        PeriodicThreadRegistry.get_instance().unregister(self.name)

        # Use base class stop method
        self.stop()
        logger.info("PinMonitor stopped")

    def get_monitored_count(self) -> int:
        """Get the number of currently monitored pinned objects."""
        with self._objects_lock:
            return len(self._pinned_objects)

    @staticmethod
    def DestroyInstance():
        """Destroy the singleton instance and stop monitoring.
        This is mainly used for testing to ensure clean state between tests.
        """
        with PinMonitor._instance_lock:
            if PinMonitor._instance is not None:
                PinMonitor._instance.stop_monitoring()
                PinMonitor._instance = None
