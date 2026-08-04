# SPDX-License-Identifier: Apache-2.0
"""
Base classes for health monitoring.
"""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.health_monitor.constants import (
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_PING_INTERVAL,
    FallbackPolicy,
)
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
)
from lmcache.v1.utils.subclass_discovery import discover_subclasses

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)


class HealthCheck(ABC):
    """
    Abstract base class for health checks.

    Subclasses should implement the check() method to perform specific
    health checks. Each health check represents one aspect of system health.

    Subclasses must also implement the create() classmethod
    to create instances from a LMCacheManager.

    Attributes:
        fallback_policy: The fallback policy when this health check fails.
            - RECOMPUTE: Skip all cache operations, fall back to recomputation
            - LOCAL_CPU: Fall back to local CPU backend only

    Example:
        class DatabaseHealthCheck(HealthCheck):
            def __init__(self, db_connection, fallback_policy=FallbackPolicy.RECOMPUTE):
                self._fallback_policy = fallback_policy
                self.db = db_connection

            def name(self) -> str:
                return "DatabaseHealthCheck"

            def check(self) -> bool:
                return self.db.ping()

            @property
            def fallback_policy(self) -> FallbackPolicy:
                return self._fallback_policy

            @classmethod
            def create(
                cls, manager: "LMCacheManager"
            ) -> List["HealthCheck"]:
                # Create instances from manager's components
                return [cls(manager.lmcache_engine.db_connection)]
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this health check"""
        pass

    @abstractmethod
    def check(self) -> bool:
        """
        Perform the health check.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    @property
    def fallback_policy(self) -> FallbackPolicy:
        """
        Return the fallback policy for this health check.

        Default is RECOMPUTE. Subclasses can override this property
        to return a different policy.

        Returns:
            FallbackPolicy: The fallback policy
        """
        return DEFAULT_FALLBACK_POLICY

    def should_skip(self) -> bool:
        """
        Check if this health check should be skipped.

        Override this method to conditionally skip checks
        (e.g., if the component doesn't support health checks).

        Returns:
            bool: True if the check should be skipped, False otherwise
        """
        return False

    def get_bypass_backend_name(self) -> Optional[str]:
        """
        Return the backend name to bypass when this health check fails
        and fallback_policy is LOCAL_CPU.

        Override this method to specify the backend name.

        Returns:
            Optional[str]: The backend name to bypass, or None if not applicable
        """
        return None

    @classmethod
    @abstractmethod
    def create(cls, manager: "LMCacheManager") -> List["HealthCheck"]:
        """
        Create health check instance(s) from a LMCacheManager.

        This method should extract the necessary components from the manager
        and create one or more health check instances.

        Args:
            manager: The LMCacheManager instance

        Returns:
            List[HealthCheck]: List of health check instances.
                              Return empty list if the check is not applicable.
        """
        pass


class HealthMonitor(PeriodicThread):
    """
    Health monitor for the entire LMCache system.

    This is the unified health monitor for the entire LMCache system.
    It supports extensible health checks and provides a centralized way
    to check the health status of the LMCacheManager.

    The monitor automatically discovers and instantiates all HealthCheck
    subclasses using their create() method.

    The monitor runs in a background thread and periodically executes
    all registered health checks. If any check fails, the system is
    marked as unhealthy and appropriate fallback actions are taken
    based on each check's fallback_policy.

    Fallback Policies:
        - RECOMPUTE: Mark system as unhealthy, skip all cache operations
        - LOCAL_CPU: Bypass the failed backend, use local CPU with hot_cache enabled

    Usage:
        # Create a health monitor with manager
        health_monitor = HealthMonitor(
            manager=manager,
            ping_interval=30.0
        )

        # Start monitoring
        health_monitor.start()

        # Check health status
        if health_monitor.is_healthy():
            # Perform normal operations
            pass

        # Stop monitoring when done
        health_monitor.stop()
    """

    def __init__(
        self,
        manager: "LMCacheManager",
        ping_interval: float = DEFAULT_PING_INTERVAL,
    ):
        # Initialize PeriodicThread base class
        super().__init__(
            name="health-monitor-thread",
            interval=ping_interval,
            level=ThreadLevel.CRITICAL,
            init_wait=0.0,
        )

        self._manager = manager
        self._health_checks: List[HealthCheck] = []

        # Health status
        self._healthy = True
        self._health_lock = threading.RLock()

        # Configuration (also stored in base class)
        self._ping_interval = ping_interval

        # Track which backends are currently bypassed due to health check failures
        # Key: backend_name, Value: check_name that caused the bypass
        self._bypassed_backends: Dict[str, str] = {}
        self._bypass_lock = threading.RLock()

        # Track original hot_cache setting before any LOCAL_CPU fallback
        # This is a global setting, not per-backend
        # None means no fallback is active, otherwise stores the original use_hot value
        self._original_hot_cache: Optional[bool] = None

        # Track previous health status per check for detecting recovery
        self._previous_check_status: Dict[str, bool] = {}

        # Auto-discover and instantiate health checks
        self._discover_health_checks()

        # Register with the global registry
        PeriodicThreadRegistry.get_instance().register(self)

    def _discover_health_checks(self) -> None:
        """
        Discover all HealthCheck subclasses and instantiate them.

        This method dynamically scans all modules in the checks package,
        finds all HealthCheck subclasses and calls their `create()`
        method to create instances.
        """
        cls: type[HealthCheck]
        for cls in discover_subclasses(
            "lmcache.v1.health_monitor.checks",
            HealthCheck,  # type: ignore[type-abstract]
            module_filter=lambda name: not name.startswith("_"),
            include_abstract=True,
            require_defined_in_module=False,
        ):
            try:
                instances = cls.create(self._manager)
            except Exception as e:
                logger.warning(f"Failed to create health check {cls.__name__}: {e}")
                continue
            for instance in instances:
                self._health_checks.append(instance)
                # Initialize previous status as healthy
                self._previous_check_status[instance.name()] = True
                logger.info(
                    f"Registered health check: {instance.name()} "
                    f"with fallback_policy: {instance.fallback_policy}"
                )

    def get_health_checks(self) -> List[HealthCheck]:
        """Get all registered health checks"""
        return list(self._health_checks)

    def is_healthy(self) -> bool:
        """
        Check if the system is currently healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        with self._health_lock:
            return self._healthy

    def _set_healthy(self, healthy: bool) -> None:
        """Set the health status"""
        with self._health_lock:
            if self._healthy != healthy:
                if healthy:
                    logger.info(
                        "HealthMonitor: System recovered, restoring normal operations"
                    )
                else:
                    logger.warning(
                        "HealthMonitor: System unhealthy, entering degraded mode"
                    )
                self._healthy = healthy

    def _get_storage_manager(self) -> Optional["StorageManager"]:
        """Get the storage manager from the cache engine."""
        engine = self._manager.lmcache_engine
        if engine is None:
            return None
        return engine.storage_manager

    def _get_local_cpu_backend(
        self, storage_manager: "StorageManager"
    ) -> Optional["LocalCPUBackend"]:
        """
        Get the LocalCPUBackend from the storage manager with proper type.

        Args:
            storage_manager: The storage manager instance

        Returns:
            Optional[LocalCPUBackend]: The LocalCPUBackend instance, or None
        """
        # First Party
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

        backend = storage_manager.local_cpu_backend
        if backend is not None and isinstance(backend, LocalCPUBackend):
            return backend
        return None

    def _apply_local_cpu_fallback(self, check: HealthCheck) -> None:
        """
        Apply LOCAL_CPU fallback policy for a failed health check.

        This will:
        1. Enable bypass for the specified backend in StorageManager
        2. Enable hot_cache for LocalCPUBackend (only on first fallback)

        Args:
            check: The health check that failed
        """
        backend_name = check.get_bypass_backend_name()
        if backend_name is None:
            logger.warning(
                f"Health check {check.name()} has LOCAL_CPU fallback but "
                "get_bypass_backend_name() returned None"
            )
            return

        storage_manager = self._get_storage_manager()
        if storage_manager is None:
            logger.warning("StorageManager is not available for fallback")
            return

        with self._bypass_lock:
            if backend_name in self._bypassed_backends:
                # Already bypassed
                return

            local_cpu = self._get_local_cpu_backend(storage_manager)

            # Save original hot_cache setting only on the first fallback
            # (when no backends are bypassed yet)
            if len(self._bypassed_backends) == 0:
                if local_cpu is not None:
                    self._original_hot_cache = local_cpu.use_hot
                else:
                    self._original_hot_cache = None

            # Enable bypass for the backend
            storage_manager.set_backend_bypass(backend_name, True)

            # Enable hot_cache for LocalCPUBackend
            if local_cpu is not None:
                local_cpu.use_hot = True
                logger.info(
                    f"Enabled hot_cache for LocalCPUBackend due to "
                    f"{check.name()} failure"
                )

            # Record this backend as bypassed
            self._bypassed_backends[backend_name] = check.name()

            logger.info(
                f"Applied LOCAL_CPU fallback for {check.name()}: "
                f"bypassing {backend_name}"
            )

    def _recover_from_local_cpu_fallback(self, check: HealthCheck) -> None:
        """
        Recover from LOCAL_CPU fallback when a health check passes again.

        This will:
        1. Disable bypass for the specified backend in StorageManager
        2. Only when ALL backends have recovered:
           - Restore original hot_cache setting for LocalCPUBackend
           - Clear hot_cache if it was originally disabled

        Args:
            check: The health check that recovered
        """
        backend_name = check.get_bypass_backend_name()
        if backend_name is None:
            return

        storage_manager = self._get_storage_manager()
        if storage_manager is None:
            return

        with self._bypass_lock:
            if backend_name not in self._bypassed_backends:
                # Not in bypassed state
                return

            check_name = self._bypassed_backends[backend_name]

            # Verify this is the same check that caused the bypass
            if check_name != check.name():
                return

            # Disable bypass for the backend
            storage_manager.set_backend_bypass(backend_name, False)

            # Remove from bypassed backends
            del self._bypassed_backends[backend_name]

            logger.info(
                f"Recovered from LOCAL_CPU fallback for {check.name()}: "
                f"restored {backend_name}"
            )

            # Only restore hot_cache when ALL backends have recovered
            if len(self._bypassed_backends) == 0:
                local_cpu = self._get_local_cpu_backend(storage_manager)
                if local_cpu is not None and self._original_hot_cache is not None:
                    # First, restore the original hot_cache setting
                    # This prevents new data from being written during clear()
                    local_cpu.use_hot = self._original_hot_cache
                    logger.info(
                        f"Restored hot_cache setting to {self._original_hot_cache} "
                        f"for LocalCPUBackend (all backends restored)"
                    )

                    # Then, clear hot_cache if it was originally disabled
                    # At this point, use_hot is already False, so no new data
                    # will be written during the potentially long clear() operation
                    if not self._original_hot_cache:
                        local_cpu.clear()
                        logger.info(
                            "Cleared hot_cache for LocalCPUBackend during recovery "
                            "(all backends restored)"
                        )
                # Reset original_hot_cache tracker
                self._original_hot_cache = None

    def _run_all_checks(self) -> bool:
        """
        Run all health checks.

        Returns:
            bool: True if all checks pass (considering fallback policies),
                  False if any check with RECOMPUTE policy fails

        Raises:
            IrrecoverableException: If any check raises an irrecoverable error
        """
        all_healthy = True

        for check in self._health_checks:
            if check.should_skip():
                continue

            check_name = check.name()
            was_healthy = self._previous_check_status.get(check_name, True)

            try:
                is_healthy = check.check()
            except IrrecoverableException:
                logger.error(f"Health check {check_name} raised IrrecoverableException")
                raise
            except Exception as e:
                logger.error(f"Health check {check_name} raised exception: {e}")
                is_healthy = False

            # Update previous status
            self._previous_check_status[check_name] = is_healthy

            if is_healthy:
                # Check recovered
                if not was_healthy:
                    logger.info(f"Health check {check_name} recovered")
                    # If this check was using LOCAL_CPU fallback, recover
                    if check.fallback_policy == FallbackPolicy.LOCAL_CPU:
                        self._recover_from_local_cpu_fallback(check)
            else:
                # Check failed
                logger.warning(f"Health check failed: {check_name}")

                if check.fallback_policy == FallbackPolicy.RECOMPUTE:
                    # RECOMPUTE policy: mark as unhealthy
                    all_healthy = False
                elif check.fallback_policy == FallbackPolicy.LOCAL_CPU:
                    # LOCAL_CPU policy: apply fallback
                    self._apply_local_cpu_fallback(check)
                    # System is still considered healthy with LOCAL_CPU fallback

        return all_healthy

    def start(self) -> Optional[threading.Thread]:
        """
        Start the health monitor thread.

        Returns:
            Optional[threading.Thread]: The started thread,
                or None if no checks need monitoring
        """
        # Check if we have any health checks that need to run
        active_checks = [c for c in self._health_checks if not c.should_skip()]
        if not active_checks:
            logger.info("No active health checks to monitor, skipping monitor thread")
            # Set last_summary even when not starting
            self._last_summary = ThreadRunSummary(
                success=True,
                message="No active health checks to monitor, thread not started",
                extra_info={
                    "total_checks": str(len(self._health_checks)),
                    "active_checks": "0",
                    "skipped": "true",
                },
            )
            self._level = ThreadLevel.LOW
            return None

        # Use the base class start method
        thread = super().start()
        if thread is not None:
            logger.info(
                f"Started health monitor thread with "
                f"{len(active_checks)} active checks, "
                f"interval: {self._ping_interval}s"
            )
        return thread

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the health monitor thread"""
        # Unregister from the global registry
        PeriodicThreadRegistry.get_instance().unregister(self.name)
        # Use base class stop method
        super().stop(timeout)

    def _execute(self) -> ThreadRunSummary:
        """
        Execute one health check cycle.

        This method is called by the PeriodicThread base class.

        Returns:
            ThreadRunSummary: Summary of the health check cycle
        """
        try:
            # Run all health checks
            is_healthy = self._run_all_checks()
            self._set_healthy(is_healthy)

            # Build summary
            failed_checks = [
                name
                for name, healthy in self._previous_check_status.items()
                if not healthy
            ]

            return ThreadRunSummary(
                success=is_healthy,
                message="All checks passed"
                if is_healthy
                else f"Failed checks: {failed_checks}",
                extra_info={
                    "total_checks": str(len(self._health_checks)),
                    "failed_checks": str(len(failed_checks)),
                    "bypassed_backends": str(len(self._bypassed_backends)),
                },
            )
        except IrrecoverableException as e:
            logger.error(f"Irrecoverable error in health monitor: {e}")
            self._set_healthy(False)
            # Re-raise to stop the thread
            raise
