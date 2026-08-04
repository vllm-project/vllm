# SPDX-License-Identifier: Apache-2.0
"""
Prefetch policy interface and default implementation for L2-to-L1 load decisions.

The prefetch policy decides which L2 adapter should load each key when multiple
adapters have the same key. It receives lookup results (bitmaps) from all adapters
and produces a load plan mapping each adapter to the key indices it should load.
"""

# Standard
from abc import ABC, abstractmethod

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)


class PrefetchPolicy(ABC):
    """
    Abstract interface for prefetch load-plan decisions.

    The prefetch policy is called by the PrefetchController after all L2
    adapters have completed their lookup_and_lock operations. Given the
    lookup results, it decides which adapter should load which keys.
    """

    @abstractmethod
    def select_load_plan(
        self,
        keys: list[ObjectKey],
        lookup_results: dict[int, Bitmap],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, Bitmap]:
        """
        Decide which adapter loads which keys.

        Args:
            keys: Full list of keys being prefetched from L2.
            lookup_results: Mapping from adapter index to Bitmap.
                A set bit at position i means the adapter has keys[i].
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to a bitmap telling which keys that
            the adapter should load. The returned bitmaps should not
            overlap, and the union of all returned bitmaps should be a subset
            of the union of the input bitmaps.
        """

    def select_l1_retentions(
        self,
        keys: list[ObjectKey],
    ) -> list[bool]:
        """Determine which keys to retain in L1 after prefetched
        objects are consumed.

        Called by PrefetchController just before
        ``l1_mgr.reserve_write`` to build the ``is_temporary``
        flags.  A ``True`` value means the key is retained
        (permanent); ``False`` means it is temporary and will be
        deleted after the reader finishes.

        The default implementation marks all keys as temporary
        (not retained).  Override in subclasses to implement
        hot-cache or selective-retention strategies.

        Args:
            keys: Keys about to be written into L1.

        Returns:
            A list of bools with the same length as *keys*.
            ``True`` = retain (permanent), ``False`` = temporary.
        """
        return [False] * len(keys)


# -----------------------------------------------------------------------------
# Registry: prefetch policy name -> policy class
# -----------------------------------------------------------------------------

_PREFETCH_POLICY_REGISTRY: dict[str, type[PrefetchPolicy]] = {}


def register_prefetch_policy(
    name: str,
    policy_cls: type[PrefetchPolicy],
) -> None:
    """
    Register a prefetch policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete PrefetchPolicy subclass.
    """
    if name in _PREFETCH_POLICY_REGISTRY:
        raise ValueError(f"Prefetch policy already registered: {name!r}")
    _PREFETCH_POLICY_REGISTRY[name] = policy_cls


def get_registered_prefetch_policies() -> list[str]:
    """Return the list of registered prefetch policy names."""
    return list(_PREFETCH_POLICY_REGISTRY)


def create_prefetch_policy(name: str) -> PrefetchPolicy:
    """
    Create a prefetch policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new PrefetchPolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _PREFETCH_POLICY_REGISTRY:
        known = ", ".join(sorted(_PREFETCH_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown prefetch policy {name!r}. Known: {known}")
    return _PREFETCH_POLICY_REGISTRY[name]()


class DefaultPrefetchPolicy(PrefetchPolicy):
    """
    Default prefetch policy: for each key, pick the first adapter
    (lowest index) that has it.
    """

    def select_load_plan(
        self,
        keys: list[ObjectKey],
        lookup_results: dict[int, Bitmap],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, Bitmap]:
        """
        Assign each key to the first adapter (by index) that has it.

        Args:
            keys: Full list of keys being prefetched.
            lookup_results: Adapter index -> Bitmap of lookup hits.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to key bitmaps. Each key goes
            to the lowest-indexed adapter that reported having it.
        """
        plan: dict[int, Bitmap] = {}
        global_bitmap = Bitmap(len(keys))
        for bitmap in lookup_results.values():
            global_bitmap |= bitmap

        for ad in sorted(adapters, key=lambda a: a.index):
            curr_bitmap = lookup_results.get(ad.index)
            if curr_bitmap is None:
                continue

            local_bitmap = global_bitmap & curr_bitmap
            global_bitmap &= ~local_bitmap
            if local_bitmap.popcount() == 0:
                continue

            plan[ad.index] = local_bitmap

        return plan


class RetainPrefetchPolicy(DefaultPrefetchPolicy):
    """Prefetch policy that retains all prefetched keys in L1.

    Inherits ``select_load_plan`` from ``DefaultPrefetchPolicy``
    (first-adapter-wins) and only overrides the L1 retention
    decision: all prefetched keys become permanent.

    Use this when prefetched data is likely to be reused by
    subsequent requests (e.g. shared system-prompt chunks).
    """

    def select_l1_retentions(
        self,
        keys: list[ObjectKey],
    ) -> list[bool]:
        """Retain all prefetched keys permanently in L1."""
        return [True] * len(keys)


register_prefetch_policy("default", DefaultPrefetchPolicy)
register_prefetch_policy("retain", RetainPrefetchPolicy)
