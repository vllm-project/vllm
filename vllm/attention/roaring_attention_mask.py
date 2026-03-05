"""
RoaringBitmap-based sparse attention mask implementation for vLLM.

This module provides memory-efficient storage for sparse attention patterns,
particularly beneficial for long-context models using sliding window or 
blockwise sparse attention.
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import torch

try:
    import pyroaring
    HAS_ROARING = True
except ImportError:
    HAS_ROARING = False


class RoaringAttentionMask:
    """
    Manages sparse attention masks using RoaringBitmaps for memory efficiency.
    
    This class is designed for attention patterns with structured sparsity
    (e.g., sliding windows, global tokens, block-local patterns).
    
    Attributes:
        num_blocks: Total number of blocks in the sequence
        masks: List of RoaringBitmaps, one per query block
        device: Target device for tensor operations
    """
    
    def __init__(self, num_blocks: int, device: str = "cpu"):
        """
        Initialize RoaringAttentionMask.
        
        Args:
            num_blocks: Total number of blocks in the sequence
            device: Device for tensor operations ("cpu" or "cuda")
        
        Raises:
            ImportError: If pyroaring is not installed
        """
        if not HAS_ROARING:
            raise ImportError(
                "RoaringAttentionMask requires pyroaring. "
                "Install with: pip install pyroaring"
            )
        
        self.num_blocks = num_blocks
        self.device = device
        self.masks = [pyroaring.BitMap() for _ in range(num_blocks)]
        
    def add_attention(self, query_block: int, key_blocks: List[int]) -> None:
        """
        Add attention connections from query_block to key_blocks.
        
        Args:
            query_block: Source block index
            key_blocks: List of target block indices
        """
        if query_block >= self.num_blocks:
            raise ValueError(f"query_block {query_block} >= num_blocks {self.num_blocks}")
        
        # Filter out invalid key blocks
        valid_keys = [k for k in key_blocks if 0 <= k < self.num_blocks]
        if valid_keys:
            self.masks[query_block].update(valid_keys)
    
    def remove_attention(self, query_block: int, key_blocks: List[int]) -> None:
        """
        Remove attention connections from query_block to key_blocks.
        
        Args:
            query_block: Source block index
            key_blocks: List of target block indices to remove
        """
        if query_block >= self.num_blocks:
            raise ValueError(f"query_block {query_block} >= num_blocks {self.num_blocks}")
        
        for key_block in key_blocks:
            self.masks[query_block].discard(key_block)
    
    def get_active_blocks(self, query_block: int) -> np.ndarray:
        """
        Get all blocks that query_block attends to.
        
        Args:
            query_block: Query block index
            
        Returns:
            Sorted array of block indices
        """
        if query_block >= self.num_blocks:
            raise ValueError(f"query_block {query_block} >= num_blocks {self.num_blocks}")
        
        return np.array(self.masks[query_block].to_array(), dtype=np.int32)
    
    def get_active_blocks_tensor(self, query_block: int) -> torch.Tensor:
        """
        Get active blocks as a PyTorch tensor.
        
        Args:
            query_block: Query block index
            
        Returns:
            Tensor of active block indices
        """
        active = self.get_active_blocks(query_block)
        return torch.tensor(active, dtype=torch.int32, device=self.device)
    
    def intersection(self, block1: int, block2: int) -> np.ndarray:
        """
        Get blocks that both block1 and block2 attend to.
        
        Args:
            block1: First block index
            block2: Second block index
            
        Returns:
            Sorted array of shared block indices
        """
        if block1 >= self.num_blocks or block2 >= self.num_blocks:
            raise ValueError(f"Block indices must be < {self.num_blocks}")
        
        intersection = self.masks[block1] & self.masks[block2]
        return np.array(intersection.to_array(), dtype=np.int32)
    
    def union(self, block1: int, block2: int) -> np.ndarray:
        """
        Get all blocks that either block1 or block2 attend to.
        
        Args:
            block1: First block index
            block2: Second block index
            
        Returns:
            Sorted array of all block indices
        """
        if block1 >= self.num_blocks or block2 >= self.num_blocks:
            raise ValueError(f"Block indices must be < {self.num_blocks}")
        
        union = self.masks[block1] | self.masks[block2]
        return np.array(union.to_array(), dtype=np.int32)
    
    def batch_intersection(self, blocks: List[int]) -> np.ndarray:
        """
        Get blocks that ALL specified blocks attend to.
        
        Args:
            blocks: List of block indices
            
        Returns:
            Sorted array of shared block indices
        """
        if not blocks:
            return np.array([], dtype=np.int32)
        
        result = self.masks[blocks[0]].copy()
        for block_idx in blocks[1:]:
            if block_idx < self.num_blocks:
                result &= self.masks[block_idx]
        
        return np.array(result.to_array(), dtype=np.int32)
    
    def has_attention(self, query_block: int, key_block: int) -> bool:
        """
        Check if query_block attends to key_block.
        
        Args:
            query_block: Query block index
            key_block: Key block index
            
        Returns:
            True if attention exists
        """
        if query_block >= self.num_blocks:
            return False
        return key_block in self.masks[query_block]
    
    def get_density(self) -> float:
        """
        Calculate the overall density of the attention mask.
        
        Returns:
            Density as a fraction (0.0 to 1.0)
        """
        total_connections = sum(len(mask) for mask in self.masks)
        total_possible = self.num_blocks * self.num_blocks
        return total_connections / total_possible if total_possible > 0 else 0.0
    
    def get_memory_usage_bytes(self) -> int:
        """
        Estimate memory usage in bytes.
        
        Returns:
            Approximate memory usage
        """
        total = 0
        for mask in self.masks:
            # Estimate: 4 bytes per value + container overhead
            num_values = len(mask)
            if num_values > 0:
                # Number of containers (65536 values per container)
                num_containers = (max(mask) // 65536) + 1 if num_values > 0 else 0
                total += num_values * 4 + num_containers * 64
            else:
                total += 64  # Empty bitmap overhead
        return total
    
    def to_dense_mask(self, dtype=torch.bool) -> torch.Tensor:
        """
        Convert to dense boolean mask (for debugging/comparison).
        
        Warning: This can use significant memory for large num_blocks.
        
        Args:
            dtype: PyTorch dtype for the mask
            
        Returns:
            Dense attention mask tensor
        """
        mask = torch.zeros(
            (self.num_blocks, self.num_blocks),
            dtype=dtype,
            device=self.device
        )
        
        for query_idx in range(self.num_blocks):
            key_indices = self.get_active_blocks(query_idx)
            if len(key_indices) > 0:
                mask[query_idx, key_indices] = True
        
        return mask
    
    def apply_sliding_window(
        self,
        window_size: int,
        global_tokens: Optional[int] = None
    ) -> None:
        """
        Apply a sliding window attention pattern.
        
        Args:
            window_size: Size of the sliding window
            global_tokens: Optional number of global attention tokens
        """
        for query_idx in range(self.num_blocks):
            key_blocks = []
            
            # Sliding window
            start = max(0, query_idx - window_size // 2)
            end = min(self.num_blocks, query_idx + window_size // 2 + 1)
            key_blocks.extend(range(start, end))
            
            # Global tokens (bidirectional)
            if global_tokens:
                if query_idx < global_tokens:
                    # Global tokens attend to all
                    key_blocks.extend(range(self.num_blocks))
                else:
                    # All blocks attend to global tokens
                    key_blocks.extend(range(global_tokens))
            
            self.add_attention(query_idx, key_blocks)
    
    def serialize(self) -> bytes:
        """
        Serialize all masks to bytes for storage/transfer.
        
        Returns:
            Serialized representation
        """
        import pickle
        serialized_masks = [mask.serialize() for mask in self.masks]
        return pickle.dumps({
            'num_blocks': self.num_blocks,
            'masks': serialized_masks
        })
    
    @classmethod
    def deserialize(cls, data: bytes, device: str = "cpu") -> "RoaringAttentionMask":
        """
        Deserialize from bytes.
        
        Args:
            data: Serialized data
            device: Target device
            
        Returns:
            RoaringAttentionMask instance
        """
        import pickle
        loaded = pickle.loads(data)
        
        instance = cls(loaded['num_blocks'], device)
        instance.masks = [
            pyroaring.BitMap.deserialize(mask_data)
            for mask_data in loaded['masks']
        ]
        return instance


def create_sliding_window_mask(
    num_blocks: int,
    window_size: int,
    global_tokens: int = 0,
    device: str = "cpu"
) -> RoaringAttentionMask:
    """
    Create a sliding window attention mask with optional global tokens.
    
    Args:
        num_blocks: Total number of blocks
        window_size: Size of sliding window
        global_tokens: Number of global attention tokens
        device: Target device
        
    Returns:
        Configured RoaringAttentionMask
    """
    mask = RoaringAttentionMask(num_blocks, device)
    mask.apply_sliding_window(window_size, global_tokens)
    return mask