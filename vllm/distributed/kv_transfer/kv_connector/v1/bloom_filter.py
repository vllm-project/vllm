# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bloom filter for decentralized KV cache block discovery.

Each vLLM instance maintains a local bloom filter summarizing its cached
block hashes, and a merged peer bloom filter (bitwise OR of all peers'
local filters) used to determine whether a peer has a needed KV block.

Based on the OfflineState design from "Cuckoo for Clients: Disaggregated
Cuckoo Hashing" thesis, adapted for vLLM's KV cache block discovery.

Key properties:
- No false negatives: if a block is cached, the filter will say "maybe"
- Bounded false positives: tunable via fp_rate parameter
- Auto-scaling: bloom size adapts to cluster size via sqrt(N) formula
- Compact: ~12 KB per node for 10K blocks at 1% FPR (packed bit storage)
"""

import math
from collections.abc import Iterable

import numpy as np


class BloomFilter:
    """Numpy-backed bloom filter with murmurhash-style integer hashing.

    Optimized for KV cache block hash lookups where keys are 64-bit
    integers (block hashes from vLLM's hash_block_tokens).
    """

    def __init__(
        self,
        expected_items: int = 10000,
        fp_rate: float = 0.01,
        auto_scale_clients: int = 1,
    ):
        """Initialize bloom filter with optimal sizing.

        Args:
            expected_items: Expected number of items per node.
            fp_rate: Target false positive rate (0.0 to 1.0).
            auto_scale_clients: Number of clients for auto-scaling.
                Bloom size scales by ceil(sqrt(auto_scale_clients))
                to keep FPR bounded when filters are merged via OR.
        """
        # Auto-scale: merged bloom of N clients has higher FPR;
        # scale expected items by sqrt(N) to compensate
        scale = max(1, math.ceil(math.sqrt(auto_scale_clients)))
        scaled_items = max(1, expected_items * scale)

        # Optimal bloom filter sizing:
        # m = -(n * ln(p)) / (ln(2)^2)
        # k = (m/n) * ln(2)
        ln2_sq = math.log(2) ** 2
        self._size_bits = max(64, int(-scaled_items * math.log(fp_rate) / ln2_sq))
        self._num_hashes = max(1, int((self._size_bits / scaled_items) * math.log(2)))

        # Packed bit storage: each byte holds 8 bits
        self._num_bytes = (self._size_bits + 7) // 8
        self._bits = np.zeros(self._num_bytes, dtype=np.uint8)
        self._item_count = 0

        # Deterministic seeds for hash functions
        self._seeds = np.array(
            [0xDEADBEEF + i * 0x9E3779B9 for i in range(self._num_hashes)],
            dtype=np.uint64,
        )

    @property
    def size_bits(self) -> int:
        """Total number of bits in the filter."""
        return self._size_bits

    @property
    def num_hashes(self) -> int:
        """Number of hash functions."""
        return self._num_hashes

    @property
    def item_count(self) -> int:
        """Number of items added (approximate after rebuild)."""
        return self._item_count

    @property
    def size_bytes(self) -> int:
        """Size of the packed bit array in bytes."""
        return self._num_bytes

    @property
    def fill_rate(self) -> float:
        """Fraction of bits set (saturation indicator)."""
        set_bits = sum(int(b).bit_count() for b in self._bits)
        return set_bits / self._size_bits

    def _hash(self, key: int, seed: int) -> int:
        """Murmurhash-style 64-bit integer finalizer."""
        h = (key ^ seed) & 0xFFFFFFFFFFFFFFFF
        h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
        h = h ^ (h >> 33)
        return h % self._size_bits

    def _set_bit(self, bit_idx: int) -> None:
        """Set a single bit in the packed array."""
        self._bits[bit_idx >> 3] |= np.uint8(1 << (bit_idx & 7))

    def _get_bit(self, bit_idx: int) -> bool:
        """Get a single bit from the packed array."""
        return bool(self._bits[bit_idx >> 3] & np.uint8(1 << (bit_idx & 7)))

    def add(self, key: int) -> None:
        """Add a block hash to the filter."""
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            self._set_bit(idx)
        self._item_count += 1

    def contains(self, key: int) -> bool:
        """Check if a block hash might be in the filter.

        Returns:
            True if the key might be present (with FPR probability of
            false positive). False means definitely not present.
        """
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            if not self._get_bit(idx):
                return False
        return True

    def clear(self) -> None:
        """Clear all bits in the filter."""
        self._bits[:] = 0
        self._item_count = 0

    def rebuild_from_keys(self, keys: Iterable[int]) -> None:
        """Clear and re-add all keys.

        This handles LRU evictions — only keys currently in cache
        will be in the rebuilt filter.
        """
        self.clear()
        for k in keys:
            self.add(k)

    def copy(self) -> "BloomFilter":
        """Create a deep copy of this filter."""
        bf = BloomFilter.__new__(BloomFilter)
        bf._size_bits = self._size_bits
        bf._num_bytes = self._num_bytes
        bf._num_hashes = self._num_hashes
        bf._bits = self._bits.copy()
        bf._item_count = self._item_count
        bf._seeds = self._seeds
        return bf

    def to_bytes(self) -> bytes:
        """Serialize the filter for network transfer.

        Format: size_bits(8) + num_hashes(4) + item_count(4) + bits
        """
        header = (
            self._size_bits.to_bytes(8, "big")
            + self._num_hashes.to_bytes(4, "big")
            + self._item_count.to_bytes(4, "big")
        )
        return header + self._bits.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialize a filter from network transfer."""
        size_bits = int.from_bytes(data[:8], "big")
        num_hashes = int.from_bytes(data[8:12], "big")
        item_count = int.from_bytes(data[12:16], "big")

        bf = cls.__new__(cls)
        bf._size_bits = size_bits
        bf._num_bytes = (size_bits + 7) // 8
        bf._num_hashes = num_hashes
        bf._item_count = item_count
        bf._bits = np.frombuffer(data[16:], dtype=np.uint8).copy()
        bf._seeds = np.array(
            [0xDEADBEEF + i * 0x9E3779B9 for i in range(num_hashes)],
            dtype=np.uint64,
        )
        return bf

    @staticmethod
    def bitwise_or(filters: list["BloomFilter"]) -> "BloomFilter":
        """Compute bitwise OR of multiple bloom filters.

        Used during periodic sync to merge all peers' local filters
        into a single merged filter for discovery.

        All filters must have the same size_bits and num_hashes.
        """
        if not filters:
            raise ValueError("Need at least one filter to merge")
        result = filters[0].copy()
        for f in filters[1:]:
            np.bitwise_or(result._bits, f._bits, out=result._bits)
        result._item_count = sum(f._item_count for f in filters)
        return result

    def theoretical_fpr(self) -> float:
        """Compute theoretical false positive rate.

        Formula: p = (1 - e^(-kn/m))^k
        """
        if self._item_count == 0:
            return 0.0
        k = self._num_hashes
        n = self._item_count
        m = self._size_bits
        return (1 - math.exp(-k * n / m)) ** k

    def __repr__(self) -> str:
        return (
            f"BloomFilter(size_bits={self._size_bits}, "
            f"num_hashes={self._num_hashes}, "
            f"items={self._item_count}, "
            f"fill_rate={self.fill_rate:.3f}, "
            f"theoretical_fpr={self.theoretical_fpr():.4f})"
        )
