# SPDX-License-Identifier: Apache-2.0
# Stub for the native_storage_ops C++ extension (implemented in csrc/storage_manager/).

"""Native storage operations for LMCache."""

# Standard
from collections.abc import Sequence
from typing import Any, Set, overload

class TTLLock:
    """
    A thread-safe lock with TTL (Time-To-Live) support.

    The lock maintains a counter that can be incremented (lock) and decremented
    (unlock). If the TTL expires, the lock is considered unlocked regardless
    of the counter value.
    """

    def __init__(self, ttl_second: int = 300) -> None:
        """
        Construct a TTLLock with the specified TTL duration in seconds.

        Args:
            ttl_second: TTL duration in seconds. Default is 300.
        """
        ...

    def lock(self) -> None:
        """
        Increment the lock counter by 1 and update the TTL.
        If the previous TTL has expired, reset counter to 1.
        """
        ...

    def unlock(self) -> None:
        """Decrement the lock counter by 1 (minimum 0)."""
        ...

    def is_locked(self) -> bool:
        """
        Check if the lock is held (counter > 0 and TTL not expired).

        Returns:
            True if the lock is held, False otherwise.
        """
        ...

    def reset(self) -> None:
        """Reset the lock to initial state (counter = 0, TTL expired)."""
        ...

class Bitmap:
    """
    A bitmap for tracking the state of L2 storage operation results.

    Each bit represents the success or failure of a key.
    """

    @overload
    def __init__(self, size: int) -> None:
        """
        Construct a Bitmap with the specified number of bits.

        Args:
            size: The number of bits in the bitmap.
        """
        ...
    @overload
    def __init__(self, size: int, prefix_bits: int) -> None:
        """
        Construct a Bitmap with the specified number of bits and prefix.

        Args:
            size: The number of bits in the bitmap.
            prefix_bits: The first N bits are set to 1.
        """
        ...

    def set(self, index: int) -> None:
        """Set the bit at the specified index to 1."""
        ...

    def batched_set(self, indices: Sequence[int]) -> None:
        """Set every bit in ``indices`` to 1 (positions >= size ignored)."""
        ...

    def set_range(self, start: int, end: int) -> None:
        """Set every bit in the half-open range ``[start, end)`` to 1.

        ``end`` is clamped to the bitmap size. Fills whole bytes, so far cheaper
        than per-bit ``set`` for a contiguous span.
        """
        ...

    def clear(self, index: int) -> None:
        """Clear the bit at the specified index to 0."""
        ...

    def test(self, index: int) -> bool:
        """
        Test the bit at the specified index.

        Returns:
            True if the bit is set to 1, False otherwise.
        """
        ...

    def popcount(self) -> int:
        """Return the number of bits set to 1."""
        ...

    def count_leading_zeros(self) -> int:
        """Return the number of leading zeros."""
        ...

    def count_leading_ones(self) -> int:
        """Return the number of leading ones."""
        ...

    def highest_set_bit(self) -> int:
        """Index of the highest set bit, or ``-1`` if no bit is set.

        The value is signed so the empty bitmap is representable (an unsigned
        index could not encode it).
        """
        ...

    def __and__(self, other: Bitmap) -> Bitmap:
        """
        Bitwise AND with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        ...

    def __invert__(self) -> Bitmap:
        """Bitwise NOT (flip all bits)."""
        ...

    def __or__(self, other: Bitmap) -> Bitmap:
        """
        Bitwise OR with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        ...

    def get_indices_list(self) -> list[int]:
        """Return a list of indices where the bit is set to 1, in ascending order."""
        ...

    def get_indices_set(self) -> Set[int]:
        """Return a set of indices where the bit is set to 1."""
        ...

    def gather(self, items: Sequence[Any]) -> list[Any]:
        """
        Return elements from items at indices where the bit is set to 1.

        Args:
            items: A sequence of objects. Length should match the bitmap size.

        Returns:
            A list of objects from items at positions where the bitmap bit is 1.
        """
        ...

    def __repr__(self) -> str:
        """String representation: '1' for set bits, '0' for clear bits."""
        ...

def fold(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Fold per-(group, chunk, rank) presence into servable prefix lengths.

    Args:
        found: Group-major / chunk-major / rank-minor presence bitmap of length
            ``len(group_windows) * num_chunks * num_ranks``.
        num_chunks: Number of LMCache chunks in the request.
        num_ranks: Number of kv_rank shards per chunk.
        group_windows: Per-object-group cross-chunk window size in chunks;
            ``<= 0`` means full attention.

    Returns:
        A bitmap of size ``num_chunks + 1``; bit ``L`` set iff every object
        group can serve a length-``L`` prefix. Bit 0 is always set.
    """
    ...

def unfold(
    hit_length: int,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Expand a model-wide hit length into the per-group retain mask.

    Args:
        hit_length: Model-wide prefix hit length in chunks (clamped to
            ``num_chunks``).
        num_chunks: Number of LMCache chunks in the request.
        num_ranks: Number of kv_rank shards per chunk.
        group_windows: Per-object-group cross-chunk window size in chunks;
            ``<= 0`` means full attention.

    Returns:
        Retain mask of length ``len(group_windows) * num_chunks * num_ranks``
        (all ranks of each retained ``(group, chunk)`` set).
    """
    ...

class ParallelPatternMatcher:
    """
    Pattern matcher for integer vectors.

    This class performs pattern matching on a vector of integers.
    It finds all positions where a given pattern occurs in the input data.
    """

    def __init__(self, pattern: list[int]) -> None:
        """
        Construct a ParallelPatternMatcher with the specified pattern.

        Args:
            pattern: The pattern to search for. Must not be empty.

        Raises:
            ValueError: If pattern is empty.
        """
        ...

    def match(self, data: list[int]) -> list[int]:
        """
        Match the pattern in the given data.

        Args:
            data: The data to search in.

        Returns:
            A sorted list of positions where the pattern starts.
            Returns an empty list if no matches are found.
        """
        ...

class PeriodicEventNotifier:
    """Singleton that periodically signals registered file descriptors.

    A background thread writes to every registered fd at a configurable
    interval.  The thread sleeps when no fds are registered and wakes
    automatically when the first fd is added.
    """

    @staticmethod
    def create(interval_ms: int, use_eventfd: bool) -> None:
        """Create the singleton. Idempotent -- second call is a no-op.

        Args:
            interval_ms: Notification interval in milliseconds.
            use_eventfd: True to write as eventfd (8-byte uint64),
                False to write as pipe (1-byte).
        """
        ...

    @staticmethod
    def get() -> PeriodicEventNotifier | None:
        """Return the singleton instance, or None if not created."""
        ...

    @staticmethod
    def shutdown() -> None:
        """Shut down the singleton and join its thread. Idempotent."""
        ...

    def register_fd(self, fd: int) -> None:
        """Register a file descriptor for periodic signaling.

        Args:
            fd: The file descriptor to signal. For eventfd, pass the
                eventfd itself. For pipes, pass the write end.
        """
        ...

    def unregister_fd(self, fd: int) -> None:
        """Unregister a file descriptor. No-op if not registered.

        Args:
            fd: The file descriptor to stop signaling.
        """
        ...

    def set_interval_ms(self, interval_ms: int) -> None:
        """Change the notification interval. Clamped to >= 1ms.

        Args:
            interval_ms: New interval in milliseconds.
        """
        ...

class RangePatternMatcher:
    """
    Range pattern matcher for integer vectors.

    This class performs range pattern matching on a vector of integers.
    It finds ranges that start with a start pattern and end with an end pattern.
    When multiple end patterns exist after a start pattern, it matches the first
    one (minimal range).
    """

    def __init__(self, start_pattern: list[int], end_pattern: list[int]) -> None:
        """
        Construct a RangePatternMatcher with start and end patterns.

        Args:
            start_pattern: The pattern marking the start of a range.
            end_pattern: The pattern marking the end of a range.

        Raises:
            ValueError: If either pattern is empty or has more than 5 elements.
        """
        ...

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        """
        Match ranges in the given data.

        Finds all ranges that start with the start pattern and end with the end
        pattern. When multiple end patterns exist after a start pattern, matches
        the first one (minimal range).

        Args:
            data: The data to search in.

        Returns:
            A list of (start_pos, end_pos) tuples where:
            - start_pos is the beginning index of the start pattern
            - end_pos is the exclusive index after the end pattern
            Returns an empty list if no ranges are found.

        Example:
            >>> start = [1, 2]
            >>> end = [3, 4]
            >>> matcher = RangePatternMatcher(start, end)
            >>> data = [1, 2, 0, 3, 4, 0, 3, 4, 1, 2, 0, 0, 3, 4]
            >>> matcher.match(data)
            [(0, 5), (8, 14)]
        """
        ...
