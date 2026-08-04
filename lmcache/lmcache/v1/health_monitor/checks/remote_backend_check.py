# SPDX-License-Identifier: Apache-2.0
"""
Health check for RemoteBackend.
"""

# Standard
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional
import asyncio
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.health_monitor.base import HealthCheck
from lmcache.v1.health_monitor.constants import (
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_GET_BLOCKING_FAILED_THRESHOLD,
    DEFAULT_PING_TIMEOUT,
    DEFAULT_WAITING_TIME_FOR_RECOVERY,
    FALLBACK_POLICY_CONFIG_KEY,
    GET_BLOCKING_FAILED_THRESHOLD_CONFIG_KEY,
    PING_GENERIC_ERROR_CODE,
    PING_TIMEOUT_CONFIG_KEY,
    PING_TIMEOUT_ERROR_CODE,
    WAITING_TIME_FOR_RECOVERY_CONFIG_KEY,
    FallbackPolicy,
)
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector
from lmcache.v1.storage_backend.connector.audit_connector import AuditConnector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)


class RemoteBackendHealthCheck(HealthCheck):
    """
    Health check for RemoteBackend by pinging the remote connector.

    This check verifies that the remote backend is reachable and responsive
    by sending periodic ping requests.

    Fallback Policies:
        - RECOMPUTE (default): Skip all cache operations when
          remote backend is unhealthy
        - LOCAL_CPU: Bypass remote backend and use local CPU with hot_cache enabled
    """

    def __init__(
        self,
        backend: "RemoteBackend",
    ):
        self.backend = backend
        # Get fallback policy from config
        fallback_policy_str = backend.config.get_extra_config_value(
            FALLBACK_POLICY_CONFIG_KEY, DEFAULT_FALLBACK_POLICY.value
        )
        # Convert string to FallbackPolicy enum
        if isinstance(fallback_policy_str, str):
            try:
                self._fallback_policy = FallbackPolicy(fallback_policy_str)
            except ValueError:
                logger.warning(
                    f"Invalid fallback_policy '{fallback_policy_str}' "
                    f"for {backend}, using default: "
                    f"{DEFAULT_FALLBACK_POLICY}"
                )
                self._fallback_policy = DEFAULT_FALLBACK_POLICY
        elif isinstance(fallback_policy_str, FallbackPolicy):
            self._fallback_policy = fallback_policy_str
        self.failure_time: Optional[float] = None
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._backend_name: Optional[str] = None
        self._last_get_blocking_failed_count = 0

    @classmethod
    def create(cls, manager: "LMCacheManager") -> List[HealthCheck]:
        """
        Create RemoteBackendHealthCheck instances from a LMCacheManager.

        This method finds all RemoteBackend instances in the storage manager
        and creates a health check for each one.

        Args:
            manager: The LMCacheManager instance

        Returns:
            List of RemoteBackendHealthCheck instances
        """
        # Import here to avoid circular imports
        # First Party
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        instances: List[HealthCheck] = []

        # Get engine from manager
        engine = manager.lmcache_engine
        if engine is None or engine.storage_manager is None:
            return instances

        for backend_name, backend in engine.storage_manager.storage_backends.items():
            if isinstance(backend, RemoteBackend):
                check = cls(backend)
                check._backend_name = backend_name
                instances.append(check)
                logger.info(f"Created {check} for {backend_name}")

        return instances

    def name(self) -> str:
        return f"RemoteBackendHealthCheck({self.backend.remote_url})"

    @property
    def fallback_policy(self) -> FallbackPolicy:
        """Return the fallback policy for this health check."""
        return self._fallback_policy

    def get_bypass_backend_name(self) -> Optional[str]:
        """
        Return the backend name to bypass when this health check fails.

        Returns:
            Optional[str]: The backend name (e.g., "RemoteBackend")
        """
        return self._backend_name

    def _get_ping_timeout(self) -> float:
        """Get the ping timeout from the backend config."""
        return self.backend.config.get_extra_config_value(
            PING_TIMEOUT_CONFIG_KEY, DEFAULT_PING_TIMEOUT
        )

    def _try_reinitialize_connection(self) -> bool:
        """
        Try to reinitialize the connection if connector is None.

        Returns:
            bool: True if connection was successfully initialized, False otherwise
        """
        if self.backend.connection is not None:
            return True

        logger.warning("Connector is None, re-initializing connection.")
        self.backend.init_connection()

        return self.backend.connection is not None

    def check(self) -> bool:
        """
        Perform a health check on remote backend, which includes the following checks:

        - get_blocking/batched_get_blocking, if failed count >= threshold,
        which means check failed, update failure_time and return False.
        If failure_time is not None, wait for more than`waiting_time_for_recovery`
        seconds before resuming the check.

        - ping, if connector supports ping, send a ping request to remote connector.

        Returns:
            bool: True if all checks succeeds, False otherwise
        """
        # Try to reinitialize connection if needed
        if not self._try_reinitialize_connection():
            return False

        # At this point, connector is guaranteed to be not None
        connector = self.backend.connection
        assert connector is not None

        if self.failure_time is not None:
            waiting_time = self.backend.config.get_extra_config_value(
                WAITING_TIME_FOR_RECOVERY_CONFIG_KEY,
                DEFAULT_WAITING_TIME_FOR_RECOVERY,
            )
            if (
                time.time() - self.failure_time > waiting_time
                and self._put_and_get_check()
            ):
                # recover from get blocking failed
                logger.info(
                    "Failure time: %s, current time: %s, "
                    "recover from get blocking failed",
                    self.failure_time,
                    time.time(),
                )
                self.failure_time = None
            else:
                logger.info(
                    "Failure time: %s, current time: %s, "
                    "still in get blocking failed recovery window",
                    self.failure_time,
                    time.time(),
                )
                return False

        # Check read failed
        current_get_blocking_failed_count = self.backend.get_blocking_failed_count
        get_blocking_failed_count = (
            current_get_blocking_failed_count - self._last_get_blocking_failed_count
        )
        self._last_get_blocking_failed_count = current_get_blocking_failed_count
        threshold = self.backend.config.get_extra_config_value(
            GET_BLOCKING_FAILED_THRESHOLD_CONFIG_KEY,
            DEFAULT_GET_BLOCKING_FAILED_THRESHOLD,
        )
        if get_blocking_failed_count >= threshold:
            logger.warning(
                "Detected %s get blocking failed in interval, threshold: %s",
                get_blocking_failed_count,
                threshold,
            )
            self.failure_time = time.time()
            return False

        # If connector doesn't support ping, assume it's healthy
        if not connector.support_ping():
            return True

        # Check ping
        try:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.ping(), self.backend.loop
            )
            error_code = future.result(timeout=self._get_ping_timeout())
            latency = (time.perf_counter() - start_time) * 1000

            # Record ping latency
            self._stats_monitor.update_remote_ping_latency(latency)
            # Record error code (0 means success)
            self._stats_monitor.update_remote_ping_error_code(error_code)

            if error_code != 0:
                logger.warning(f"Ping failed with error code: {error_code}")
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning("Ping timeout")
            self._stats_monitor.update_remote_ping_error_code(PING_TIMEOUT_ERROR_CODE)
            return False
        except Exception as e:
            logger.error(f"Ping error: {e}")
            self._stats_monitor.update_remote_ping_error_code(PING_GENERIC_ERROR_CODE)
            return False

    def _put_and_get_check(self) -> bool:
        if self.backend.local_cpu_backend is None or self.backend.connection is None:
            return False

        with self._resource_manager() as (put_obj, get_obj):
            if put_obj is None:
                return False
            if get_obj is None:
                logger.warning("Get failed, the return value is None, check failed.")
                return False
            return torch.equal(put_obj.raw_tensor, get_obj.raw_tensor)

    @contextmanager
    def _resource_manager(self):
        key = CacheEngineKey(
            model_name="test",
            world_size=1,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.bfloat16,
        )
        connector = self.backend.connection
        if isinstance(connector, InstrumentedRemoteConnector):
            connector = connector.getWrappedConnector()
            if isinstance(connector, AuditConnector):
                connector = connector.real_connector
        shapes = connector.meta_shapes
        dtypes = connector.meta_dtypes
        fmt = connector.meta_fmt
        put_obj, get_obj = None, None
        try:
            # put
            put_obj = self.backend.local_cpu_backend.allocate(shapes, dtypes, fmt)
            future = self.backend.submit_put_task(key, put_obj)
            future.result(timeout=self._get_ping_timeout())
            # get
            if connector.support_batched_get():
                get_objs = self.backend.batched_get_blocking([key])
                get_obj = get_objs[0] if get_objs else None
            else:
                get_obj = self.backend.get_blocking(key)
            yield put_obj, get_obj
        except asyncio.TimeoutError:
            logger.warning("Put timeout, check failed.")
            yield None, None
        except Exception as e:
            logger.error(f"Put error, check failed: {e}")
            yield None, None
        finally:
            if put_obj is not None:
                put_obj.ref_count_down()
            if get_obj is not None:
                get_obj.ref_count_down()
