"""Block hash types for KV cache prefix caching."""
from typing import Any, NamedTuple, Optional


class BlockHash(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


class BlockHashWithGroupId(NamedTuple):
    """Block hash with KV cache group ID for multi-group cache management."""
    # The hash value for the contents (e.g., token_ids) of a block without
    # group ID. The value is the same for blocks representing the same tokens
    # but for different groups.
    block_hash: BlockHash
    # The KV cache group ID.
    group_id: int

    def get_hash_value(self) -> int:
        """Get the hash value of the block."""
        return self.block_hash.hash_value