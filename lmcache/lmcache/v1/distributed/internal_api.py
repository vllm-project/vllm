# SPDX-License-Identifier: Apache-2.0
"""
Class for distributed storage manager internal API data structures
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum

# First Party
from lmcache.v1.distributed.api import L1BackendType, ObjectKey


@dataclass(frozen=True)
class L1MemoryDesc:
    """
    Describes the L1 memory buffer registered with an external backend (e.g. Nixl).
    """

    ptr: int
    size: int
    align_bytes: int


@dataclass(frozen=True)
class L1ObjectMeta:
    """Per-object metadata published alongside keys in L1 cache events.

    Attributes:
        size_bytes: Logical byte size of the object.
        backend: The storage medium backing the object.
    """

    size_bytes: int
    backend: L1BackendType


class EventListener(ABC):  # noqa: B024
    pass


# For L1 manager event notifications
class L1ManagerListener(EventListener):
    """
    Listener for L1 manager events
    """

    @abstractmethod
    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that new keys have been reserved for read on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_read_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully read
        """
        pass

    @abstractmethod
    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been reserved for write on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully reserved
        """
        pass

    @abstractmethod
    def on_l1_keys_write_finished(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully written
        """
        pass

    @abstractmethod
    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been finished for writing
        and reserved for read on L1.

        This will only be trigger by the prefetch operation now.

        Args:
            keys (list[ObjectKey]): The keys that have been successfully
                finished for writing and reserved for read
        """
        # NOTE (ApostaC): may consider renaming this to `on_l1_keys_finish_prefetch`
        # for better clarity
        pass

    @abstractmethod
    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L1.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        pass

    @abstractmethod
    def on_l1_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed on L1.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        pass


class L2AdapterListener(EventListener):
    """Listener for L2 adapter events, analogous to L1ManagerListener."""

    @abstractmethod
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        """
        Notify the listener that keys have been successfully stored in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been stored.
            sizes (list[int]): The byte size of each stored key.
        """
        pass

    @abstractmethod
    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been accessed (lookup hit) in L2.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed.
        """
        pass

    @abstractmethod
    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        """
        Notify the listener that keys have been deleted from L2.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted.
        """
        pass


# For Eviction
class EvictionDestination(enum.Enum):
    """
    The destination of evicted objects
    """

    DISCARD = enum.auto()
    """Discard the evicted objects"""

    L2_CACHE = enum.auto()
    """Evict to L2 storage"""


@dataclass(frozen=True)
class EvictionAction:
    """
    An action to be taken for eviction
    """

    destination: EvictionDestination
    """The destination of the evicted object"""

    keys: list[ObjectKey] = field(default_factory=list)
    """The key of the object to be evicted"""


class L2StoreResult(int):
    """Immutable result of a completed L2 store task.

    Encodes both the success flag and bytes transferred in the int
    value: ``>= 0`` means success (value = bytes transferred);
    ``-1`` means failure.

    Args:
        success: Whether the store task succeeded.
        bytes_transferred: Bytes actually written to L2. Must be >= 0.

    Raises:
        ValueError: If ``bytes_transferred`` is negative.
    """

    def __new__(cls, success: bool, bytes_transferred: int) -> "L2StoreResult":
        if bytes_transferred < 0:
            raise ValueError(f"bytes_transferred must be >= 0, got {bytes_transferred}")
        return super().__new__(cls, bytes_transferred if success else -1)

    def is_successful(self) -> bool:
        """Return ``True`` when the store task succeeded."""
        return int(self) >= 0

    def bytes_transferred(self) -> int:
        """Return the number of bytes actually written, or 0 on failure."""
        value = int(self)
        return value if value >= 0 else 0


@dataclass(frozen=True)
class QuotaEntry:
    """Snapshot of a single quota registration."""

    cache_salt: str
    limit_bytes: int
