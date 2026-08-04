# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.quota_manager import QuotaManager

logger = init_logger(__name__)


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.
    """

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()
        self._last_extra_log = time.monotonic()

    def report_status(self) -> dict:
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
        }

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        """Publish a below-watermark loop tick (no eviction this cycle)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": False,
                },
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        """Publish an above-watermark loop tick (eviction policy ran)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": True,
                },
            )
        )

    def _maybe_log_memory_usage(self, used_bytes: int, total_bytes: int) -> None:
        """Emit the opt-in L1 memory usage INFO line, throttled to the interval."""
        now = time.monotonic()
        if now - self._last_extra_log < self._eviction_config.extra_logging_interval:
            return
        self._last_extra_log = now
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        logger.info(
            "L1 memory usage: %.2f/%.2f GiB (%.1f%%)",
            used_bytes / (1 << 30),
            total_bytes / (1 << 30),
            pct,
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            if self._eviction_config.extra_logging_enabled:
                self._maybe_log_memory_usage(used_bytes, total_bytes)
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        self.adapter_id = adapter_id
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When the adapter's policy sets ``support_isolation == True``
    (e.g. :class:`IsolatedLRUEvictionPolicy`), the controller consults
    the injected :class:`QuotaManager` to decide which ``cache_salt``
    buckets are over budget and evicts from each one in isolation.
    Otherwise it uses the adapter's aggregate ``usage_fraction``
    against the configured watermark — unchanged from the pre-PR5
    behavior.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        # Guards _adapter_states against concurrent runtime add/remove.
        self._states_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def add_adapter_state(self, state: L2AdapterEvictionState) -> None:
        """Register a new adapter's eviction state at runtime."""
        with self._states_lock:
            self._adapter_states.append(state)

    def remove_adapter_state(self, adapter_id: int) -> None:
        """Drop the eviction state for ``adapter_id``.

        Blocks until any in-progress eviction pass finishes (it holds the
        same lock), so the adapter is guaranteed idle here before the
        caller closes it. A no-op if the adapter has no eviction state.
        """
        with self._states_lock:
            self._adapter_states = [
                s for s in self._adapter_states if s.adapter_id != adapter_id
            ]

    def report_status(self) -> dict:
        # NOTE: ``usage.bytes_by_cache_salt`` is intentionally NOT
        # surfaced here. A deployment can have 10k+ salts, so embedding
        # the full bucket map in the status response would blow up the
        # payload. Per-salt inspection goes through the dedicated HTTP
        # quota endpoints (which pull from ``QuotaManager`` +
        # ``StorageManager.get_usage_bytes_by_cache_salt``).
        adapter_statuses = []
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            usage = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": usage.usage_fraction,
                    "total_bytes_used": usage.total_bytes_used,
                    "total_capacity_bytes": usage.total_capacity_bytes,
                    "num_cache_salt_buckets": len(usage.bytes_by_cache_salt),
                }
            )
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            # Hold the lock across the whole pass so remove_adapter_state
            # cannot detach (and the caller close) an adapter while we are
            # calling into it.
            with self._states_lock:
                for state in self._adapter_states:
                    self._check_and_evict(state)

    def _check_and_evict(self, state: L2AdapterEvictionState):
        if state.eviction_policy.support_isolation and self._quota_manager is not None:
            self._check_and_evict_by_cache_salt(state)
        else:
            self._check_and_evict_global(state)

    def _check_and_evict_global(self, state: L2AdapterEvictionState):
        """Aggregate-usage eviction (``LRU`` / ``noop``)."""
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio

        # ``usage_fraction == -1`` means the adapter doesn't support
        # usage-based eviction (no max_capacity_bytes declared), so we
        # do not trigger eviction. Adapters with ``supports_global_eviction ==
        # False`` should already have been filtered out at construction
        # time in ``StorageManager``; this check is a defensive belt.
        current_usage = state.adapter.get_usage().usage_fraction
        if current_usage < 0 or current_usage < watermark:
            logger.debug(
                "L2 usage %.2f below watermark %.2f; skipping eviction.",
                current_usage,
                watermark,
            )
            return

        logger.info(
            "L2 usage %.2f above watermark %.2f; triggering eviction.",
            current_usage,
            watermark,
        )
        actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
        for action in actions:
            self._execute_eviction_action(state.adapter, action)

    def _check_and_evict_by_cache_salt(self, state: L2AdapterEvictionState):
        """Per-``cache_salt`` eviction driven by :class:`QuotaManager`.

        For every salt with non-zero bytes, compare its usage against
        ``watermark * quota``. Salts over threshold get eviction scoped
        to their own LRU list. Salts with no quota registered have an
        effective limit of ``0`` and are therefore always over budget,
        so they get a full eviction (``effective_ratio=1.0``) — this
        enforces the allowlist rule: only registered salts retain data.

        Per-destination keys are batched across all over-budget salts
        before invoking the adapter — one ``adapter.delete(...)`` call
        per destination instead of one per (salt, destination) pair.
        Adapters with non-trivial per-call overhead (NIXL handle setup,
        FS sync, etc.) see this as a real win when many salts go over
        budget in the same cycle.
        """
        assert self._quota_manager is not None
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        usage = state.adapter.get_usage()

        # destination -> accumulated keys across all over-budget salts.
        pending: dict[EvictionDestination, list[ObjectKey]] = {}

        for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
            if user_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            # Trigger on ``>=`` to match the global branch's ``usage <
            # watermark`` short-circuit. Salts with no quota (limit=0)
            # always land here because ``user_bytes > 0 >= 0``.
            if user_bytes < watermark * limit:
                continue

            # Unregistered / zero-quota salts: wipe everything.
            # Registered salts: evict the configured ratio of their list.
            effective_ratio = 1.0 if limit == 0 else eviction_ratio
            logger.info(
                "cache_salt=%r over quota (bytes=%d, limit=%d, "
                "watermark=%.2f); evicting ratio=%.2f.",
                cache_salt,
                user_bytes,
                limit,
                watermark,
                effective_ratio,
            )
            actions = state.eviction_policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            for action in actions:
                pending.setdefault(action.destination, []).extend(action.keys)

        for destination, keys in pending.items():
            self._execute_eviction_action(
                state.adapter,
                EvictionAction(keys=keys, destination=destination),
            )

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        if action.destination == EvictionDestination.DISCARD:
            adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            adapter.delete(action.keys)

        if action.keys:
            get_event_bus().publish(
                Event(
                    event_type=EventType.L2_KEYS_EVICTED,
                    metadata={
                        "key_count": len(action.keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in action.keys
                        ),
                    },
                )
            )
