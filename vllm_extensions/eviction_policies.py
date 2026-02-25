from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import time

try:
    from torch import Tensor
except ImportError:
    Tensor = None

@dataclass
class BlockMetadata:
    block_id: int
    location: str = "gpu"  # "gpu", "cpu", "in_transit"
    last_access_time: float = 0.0
    access_count: int = 0
    ref_count: int = 1
    cumulative_attention_score: float = 0.0
    transfer_stream: Optional[int] = None # placeholder for cudaStream_t
    cpu_address: Optional[int] = None
    gpu_address: Optional[int] = None
    size_bytes: int = 0
    is_last_in_sequence: bool = False
    is_being_accessed: bool = False

class EvictionPolicy(ABC):
    @abstractmethod
    def select_victims(self, blocks: List[BlockMetadata], num_to_evict: int) -> List[BlockMetadata]:
        """Select which blocks to evict to CPU"""
        pass
    
    def update_on_access(self, block: BlockMetadata, attention_scores: Optional['Tensor'] = None):
        """Update metadata when block is accessed"""
        if attention_scores is not None:
            # Accumulate attention score (e.g. online softmax sum approximation)
            block.cumulative_attention_score += attention_scores.sum().item()
        block.last_access_time = time.time()
        block.access_count += 1

class LRUEvictionPolicy(EvictionPolicy):
    def select_victims(self, blocks: List[BlockMetadata], num_to_evict: int) -> List[BlockMetadata]:
        # Sort by last access time (ascending = oldest first)
        sorted_blocks = sorted(blocks, key=lambda b: b.last_access_time)
        return sorted_blocks[:num_to_evict]

class AttentionWeightedEvictionPolicy(EvictionPolicy):
    def select_victims(self, blocks: List[BlockMetadata], num_to_evict: int) -> List[BlockMetadata]:
        # Sort by cumulative attention score (ascending = lowest attention first)
        sorted_blocks = sorted(blocks, key=lambda b: b.cumulative_attention_score)
        return sorted_blocks[:num_to_evict]

class HybridEvictionPolicy(EvictionPolicy):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.alpha = alpha  # Weight for attention
        self.beta = beta    # Weight for recency
        self.gamma = gamma  # Weight for frequency
        
    def _compute_score(self, block: BlockMetadata, current_time: float) -> float:
        """Higher score = keep on GPU, lower score = evict"""
        attention_score = block.cumulative_attention_score
        recency_score = 1.0 / (current_time - block.last_access_time + 1e-6)
        frequency_score = block.access_count
        
        # Normalize and combine
        score = (self.alpha * attention_score + 
                 self.beta * recency_score + 
                 self.gamma * frequency_score)
        return score
    
    def select_victims(self, blocks: List[BlockMetadata], num_to_evict: int) -> List[BlockMetadata]:
        current_time = time.time()
        scored_blocks = [(b, self._compute_score(b, current_time)) for b in blocks]
        sorted_blocks = sorted(scored_blocks, key=lambda x: x[1])  # Ascending
        return [b for b, _ in sorted_blocks[:num_to_evict]]
