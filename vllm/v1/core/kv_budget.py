"""Runtime KV cache budget management for vLLM.

This module provides functionality to dynamically control the KV cache size
at runtime, enabling multiple vLLM instances to share GPU memory effectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class KVCacheBudget:
    """Represents a runtime budget for KV cache allocation.

    The budget is specified in bytes and can be converted to blocks based on
    the block size. This provides a universal unit across models with different
    block sizes, which is particularly useful for multi-model deployments.

    Args:
        target_bytes: Target KV cache size in bytes.
    """
    target_bytes: Optional[int] = None

    def normalized_blocks(self, bytes_per_block: int) -> Optional[int]:
        """Convert budget from bytes to number of blocks.

        Args:
            bytes_per_block: Size of each block in bytes.

        Returns:
            Number of blocks, or None if no budget is set.
        """
        if self.target_bytes is not None:
            return max(0, int(self.target_bytes // max(1, bytes_per_block)))
        return None
