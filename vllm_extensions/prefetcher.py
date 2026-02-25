from typing import Dict, List, Set
from dataclasses import dataclass, field
from vllm_extensions.tiered_block_manager import TieredBlockSpaceManager

@dataclass
class PrefetchStats:
    predictions: int = 0
    hits: int = 0
    misses: int = 0
    wasted: int = 0
    predicted_blocks: Set[int] = field(default_factory=set)
    
    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.predictions)
        
    def record_prediction(self, block_id: int):
        self.predictions += 1
        self.predicted_blocks.add(block_id)
        
    def record_access(self, block_id: int, was_prefetched: bool):
        if was_prefetched:
            self.hits += 1
        else:
            self.misses += 1

class SequentialPrefetcher:
    """Predict next blocks based on sequential access pattern"""
    
    def __init__(self, prefetch_distance: int = 3):
        self.prefetch_distance = prefetch_distance
        self.pending_prefetches: Dict[int, int] = {} # Dict block_id -> cudaStream_t (int mock)
        self.stats = PrefetchStats()
        
    def predict_next_blocks(self, current_block_id: int, block_table: List[int]) -> List[int]:
        """Simple sequential prediction based on position in block table"""
        try:
            current_pos = block_table.index(current_block_id)
        except ValueError:
            return []
            
        next_blocks = []
        for i in range(1, self.prefetch_distance + 1):
            if current_pos + i < len(block_table):
                next_blocks.append(block_table[current_pos + i])
                
        return next_blocks
        
    def prefetch(self, block_ids: List[int], manager: TieredBlockSpaceManager):
        """Async prefetch blocks from CPU"""
        for block_id in block_ids:
            if block_id in manager.cpu_blocks and block_id not in self.pending_prefetches:
                stream = manager.async_transfer_to_gpu(block_id)
                self.pending_prefetches[block_id] = stream
                self.stats.record_prediction(block_id)
                
    def check_ready(self, block_id: int) -> bool:
        """Check if prefetched block is ready"""
        if block_id not in self.pending_prefetches:
            return True # Not pending prefetch, assume ready or completely not fetched
            
        # In reality: check cudaStreamQuery
        # Mock completion check
        del self.pending_prefetches[block_id]
        return True
