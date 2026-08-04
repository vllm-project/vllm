# SPDX-License-Identifier: Apache-2.0
"""
Eviction module to determine what to evict from L1 and L2 caches.
"""

# Standard
from abc import abstractmethod
from collections.abc import Callable

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
    L1ManagerListener,
    L2AdapterListener,
)


class EvictionPolicy:
    """
    Pure abstract base class for eviction policies.

    Subclasses implement the LRU (or other) tracking logic via the
    on_keys_* methods and expose eviction decisions via get_eviction_actions.
    The binding to a specific cache tier (L1 or L2) is provided by
    L1EvictionPolicy and L2EvictionPolicy respectively.
    """

    @property
    def support_isolation(self) -> bool:
        """Whether this policy supports isolation eviction (e.g., per user isolation).

        When True, the eviction controller checks isolated usage (e.g., per user usage)
        and passes ``cache_salt`` to ``get_eviction_actions()`` to scope
        eviction to specific cache_salt. When False, the controller uses
        aggregate usage only.

        Default is False. Subclasses that support isolated eviction.
        (e.g., ``IsolatedLRUEvictionPolicy``) should override to return True.
        """
        return False

    @abstractmethod
    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to
                register.
        """
        pass

    @abstractmethod
    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that new keys have been created.

        Args:
            keys (list[ObjectKey]): The keys that have been created.
        """
        pass

    @abstractmethod
    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been accessed.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed.
        """
        pass

    @abstractmethod
    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Notify the eviction policy that keys have been deleted.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted.
        """
        pass

    @abstractmethod
    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """
        Get the eviction actions to evict objects from cache.

        Args:
            expected_ratio (float): A hint indicating approximately what
                fraction of tracked keys should be evicted. Value should be
                in range [0.0, 1.0]. For example, 0.1 means roughly 10% of
                keys should be evicted. This is a hint and the policy may
                return more or fewer keys.
            key_eligible_filter: An optional callable that takes an ObjectKey
                and returns True if the key is eligible for eviction. When
                provided, keys for which the filter returns False will be
                skipped. This is useful for skipping locked keys that
                cannot be deleted.
            cache_salt: When set, scope eviction to keys belonging to this
                salt only (identified by ``ObjectKey.cache_salt``). When
                None, evict globally across all salts. Only meaningful for
                policies where ``support_isolation`` is True; other policies
                ignore this parameter.

        Returns:
            list[EvictionAction]: The eviction actions to perform. Each
                action contains the keys and one eviction destination.

        Notes:
            The eviction action may not be successfully executed, or it may
            be executed asynchronously. Therefore, the eviction policy should
            not assume that the objects are evicted immediately, but it should
            use `on_keys_removed` to know when the objects are actually
            deleted.
        """
        pass


class L1EvictionPolicy(L1ManagerListener):
    """
    Bridges L1Manager lifecycle events to an EvictionPolicy instance.

    The actual eviction policy is provided via the constructor, keeping
    the policy logic decoupled from the listener interface.
    """

    def __init__(self, policy: EvictionPolicy):
        self._policy = policy

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    # L1ManagerListener implementations
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        # No-op
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        self._policy.on_keys_touched(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        # No-op
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        # TODO (ApostaC): we don't differentiate between the created keys and
        # updated keys here. Probably need to fix that by introducing a new
        # callback in L1ManagerListener or adding `mode` argument into
        # on_keys_reserved_write.
        self._policy.on_keys_created(keys)

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        self._policy.on_keys_removed(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        self._policy.on_keys_created(keys)

    def on_l1_keys_accessed(self, keys: list[ObjectKey]):
        self._policy.on_keys_touched(keys)


class L2EvictionPolicy(L2AdapterListener):
    """
    Bridges L2Adapter lifecycle events to an EvictionPolicy instance.

    The actual eviction policy is provided via the constructor, keeping
    the policy logic decoupled from the listener interface.
    """

    def __init__(self, policy: EvictionPolicy):
        self._policy = policy

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        self._policy.on_keys_created(keys)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self._policy.on_keys_touched(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self._policy.on_keys_removed(keys)
