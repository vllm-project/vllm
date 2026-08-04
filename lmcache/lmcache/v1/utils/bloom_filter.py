# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Union
import array
import hashlib
import math

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BloomFilter:
    """Bloom Filter for memory-efficient."""

    def __init__(
        self, expected_elements: int = 1000000, false_positive_rate: float = 0.01
    ):
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)
        array_size = (self.size + 31) // 32
        self.bit_array = array.array("I", [0] * array_size)
        self.item_count = 0
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return int(-(n * math.log(p)) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, item: Union[str, int]) -> list[int]:
        result = []
        if isinstance(item, int):
            for i in range(self.hash_count):
                h = hashlib.sha256(
                    item.to_bytes((item.bit_length() + 7) // 8, "big", signed=False)
                    + i.to_bytes(4, "big")
                ).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        else:
            for i in range(self.hash_count):
                h = hashlib.sha256(f"{item}_{i}".encode()).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        return result

    def add(self, item: Union[str, int]) -> None:
        for pos in self._hashes(item):
            self.bit_array[pos >> 5] |= 1 << (pos & 31)
        self.item_count += 1

    def add_batch_with_hashes_and_check(self, positions_list: list[list[int]]) -> int:
        """Add multiple sets of hash positions and return count of new items.

        Args:
            positions_list: List of lists, where each inner list contains
            hash positions for one item

        Returns:
            Number of new items that were actually added
        """
        unique_count = 0
        for positions in positions_list:
            is_new = any(
                not (self.bit_array[pos >> 5] & (1 << (pos & 31))) for pos in positions
            )
            if is_new:
                for pos in positions:
                    self.bit_array[pos >> 5] |= 1 << (pos & 31)
                unique_count += 1
        self.item_count += unique_count
        return unique_count

    def contains(self, item: Union[str, int]) -> bool:
        for pos in self._hashes(item):
            if not (self.bit_array[pos >> 5] & (1 << (pos & 31))):
                return False
        return True

    def clear(self) -> None:
        self.bit_array = array.array("I", [0] * ((self.size + 31) // 32))
        self.item_count = 0

    def get_memory_usage_bytes(self) -> int:
        return self.size // 8

    def get_bit_set(self) -> int:
        return sum(val.bit_count() for val in self.bit_array)

    def get_fill_rate(self) -> float:
        bits_set = self.get_bit_set()
        return bits_set / self.size if self.size > 0 else 0.0

    def get_statistics(self) -> dict:
        bits_set = self.get_bit_set()
        return {
            "size_mb": self.get_memory_usage_bytes() / 1024 / 1024,
            "hash_count": self.hash_count,
            "item_count": self.item_count,
            "bits_set": bits_set,
            "fill_rate": bits_set / self.size if self.size > 0 else 0.0,
            "expected_elements": self.expected_elements,
            "false_positive_rate": self.false_positive_rate,
        }
