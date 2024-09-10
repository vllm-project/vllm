import uuid
from typing import Dict, List, Tuple, Union, Callable

from vllm.sequence import Sequence
from vllm.inputs import LLMInputs
from vllm.core.block.prefix_caching_block import BlockTracker


ExpirationFuncType = Union[
    Callable[[BlockTracker, float, float], bool],  # ttl_expiration
    Callable[[float, float], bool],                # fixed_time_expiration
    Callable[[], bool]                             # manual_expiration
]

def ttl_expiration(block_tracker: BlockTracker, current_time: float, refresh_time: float) -> bool:
    return current_time > block_tracker.last_accessed + refresh_time

def fixed_time_expiration(current_time: float, expiration_time: float) -> bool:
    return current_time > expiration_time

def manual_expiration() -> bool:
    return False


class PinnedCachingManager:
    """Manages pinned cached sequences and their associated blocks.

    The 'PinnedCachingManager' is responsible for handling sequences and their associated blocks
    that are "pinned" in the cache. Pinned caching is an extension of prefix caching, designed to
    ensure that specific sequences remain in the cache for a longer period and have a higher
    priority for caching compared to other blocks. Pinned sequences are protected from eviction
    under certain conditions.

    Each sequence in the cache is associated with a specific number of blocks, and
    these blocks may expire based on one of three expiration strategies:
        1. TTL (Time-to-Live) expiration: The sequence expires after a specified period of inactivity.
        2. Fixed-time expiration: The sequence expires at a predetermined point in time.
        3. Manual expiration: The sequence expires only when manually triggered.

    NOTE: Pinned caching is designed to work with GPU blocks and is only compatible with prefill mode
        (max_tokens=1) to prevent data from being swapped to CPU memory.

    Args:
        refresh_time (float): The duration for which cached sequences remain valid under TTL expiration.
        max_blocks (int): The maximum number of blocks that can be stored in the cache, not considering duplication.
    """

    def __init__(
        self,
        refresh_time: float,
        max_blocks: int
    ) -> None:
        self.refresh_time = refresh_time
        self.max_blocks = max_blocks
        self.total_cached_blocks = 0
        self._cached_seqs: Dict[int, Tuple[Sequence, int, ExpirationFuncType, Tuple]] = {}
        self._expired_seqs: List[Sequence] = []

    def generate_uuid_int(self) -> int:
        return uuid.uuid4().int

    def add_seq(
        self,
        seq: Sequence,
        num_blocks: int,
        expiration_func: ExpirationFuncType,
        *args
    ) -> None:
        if self.total_cached_blocks + num_blocks > self.max_blocks:
            raise RuntimeError(f"Exceeded maximum number of pinned caching blocks ({self.max_blocks}). "
                               "Blocking further pinned caching operations.")

        attempts = 0
        max_attempts = 50
        cache_uuid = self.generate_uuid_int()

        # Ensure unique UUID.
        while cache_uuid in self._cached_seqs:
            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError(f"Failed to generate a unique UUID for pinned caching after {max_attempts} attempts")
            cache_uuid = self.generate_uuid_int()

        self._cached_seqs[cache_uuid] = (seq, num_blocks, expiration_func, args)
        self.total_cached_blocks += num_blocks

    def add_ttl_expiration_seq(self, seq: Sequence, num_blocks: int, block_tracker: BlockTracker) -> None:
        self.add_seq(seq, num_blocks, ttl_expiration, (block_tracker, self.refresh_time))

    def add_fixed_time_expiration_seq(self, seq: Sequence, num_blocks: int,  expiration_time: float) -> None:
        self.add_seq(seq, num_blocks, fixed_time_expiration, expiration_time)

    def add_manual_expiration_seq(self, seq: Sequence, num_blocks: int) -> None:
        self.add_seq(seq, num_blocks, manual_expiration)

    def manually_expire_seq(
        self,
        cache_uuid: int,
    ) -> None:
        if cache_uuid in self._cached_seqs:
            seq, num_blocks, _, _ = self._cached_seqs.pop(cache_uuid)
            self._expired_seqs.append(seq)
            self.total_cached_blocks -= num_blocks

    def expire_seqs(self, current_time: float) -> None:
        to_remove = []
        for cache_uuid, (_, num_blocks, expiration_func, args) in self._cached_seqs.items():
            if expiration_func(*args, current_time):
                to_remove.append(cache_uuid)
                self.total_cached_blocks -= num_blocks

        for uuid in to_remove:
            self._expired_seqs.append(self._cached_seqs.pop(uuid)[0])

    def get_uuid_and_inputs(
        self,
    ) -> List[Tuple[int, LLMInputs]]:
        uuid_and_inputs = []
        for cache_uuid, (seq, _, _, _) in self._cached_seqs.items():
            if seq.inputs is not None:
                uuid_and_inputs.append((cache_uuid, seq.inputs))
        return uuid_and_inputs

    def get_and_reset_expired_seq(
        self,
    ) -> List[Sequence]:
        expired_seqs = self._expired_seqs
        self._expired_seqs = []
        return expired_seqs
