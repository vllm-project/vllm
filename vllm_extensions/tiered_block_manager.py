from typing import Dict, List, Literal, Optional
import time
import torch

from vllm_extensions.eviction_policies import EvictionPolicy, BlockMetadata, LRUEvictionPolicy
from vllm_extensions.instrumentation import EvictionStats

class PhysicalTokenBlock:
    """Mock for vLLM PhysicalTokenBlock with actual tensor"""
    def __init__(self, block_id: int):
        self.block_id = block_id
        # We assume some tensor representation for the block size later
        self.tensor = None

class CPUBlockAllocator:
    def __init__(self, num_blocks: int, block_size_bytes: int = 1048576):
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes
        self.free_blocks = set(range(num_blocks))
        import torch
        # Allocate pinned CPU memory
        self.pinned_memory_pool = [
            torch.empty(block_size_bytes, dtype=torch.uint8, pin_memory=True) 
            for _ in range(num_blocks)
        ]
        
    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise RuntimeError("Out of CPU memory")
        block_id = self.free_blocks.pop()
        block = PhysicalTokenBlock(block_id=block_id)
        block.tensor = self.pinned_memory_pool[block_id]
        return block
        
    def free(self, block: PhysicalTokenBlock):
        self.free_blocks.add(block.block_id)


class BlockSpaceManagerProxy:
    """
    Mock base class proxy for vLLM BlockSpaceManager.
    In actual vLLM integration, TieredBlockSpaceManager will inherit directly from BlockSpaceManager.
    """
    def __init__(self, num_gpu_blocks: int = 100):
        self.num_gpu_blocks = num_gpu_blocks
        self.allocated_gpu_blocks = 0
        self.next_block_id = 0
        self.gpu_blocks: Dict[int, BlockMetadata] = {}
        
    def can_allocate(self) -> bool:
        return self.allocated_gpu_blocks < self.num_gpu_blocks
        
    def allocate(self, request_id: str) -> PhysicalTokenBlock:
        if not self.can_allocate():
            raise RuntimeError("GPU Full")
        
        block_id = self.next_block_id
        self.next_block_id += 1
        self.allocated_gpu_blocks += 1
        
        block = PhysicalTokenBlock(block_id=block_id)
        import torch
        # Mock GPU memory tensor
        block.tensor = torch.empty(1048576, dtype=torch.uint8, device="cuda")
        
        meta = BlockMetadata(block_id=block_id)
        meta.gpu_address = block_id
        meta.location = "gpu"
        meta.last_access_time = time.time()
        self.gpu_blocks[block_id] = meta
        
        return block
        
    def free_gpu_block(self, block: BlockMetadata):
        if block.block_id in self.gpu_blocks:
            del self.gpu_blocks[block.block_id]
            self.allocated_gpu_blocks -= 1
            
    def allocate_gpu_block(self) -> PhysicalTokenBlock:
        return self.allocate("internal_fetch")

class TieredBlockSpaceManager(BlockSpaceManagerProxy):
    def __init__(self, num_gpu_blocks: int = 100, num_cpu_blocks: int = 1000, eviction_policy: Optional[EvictionPolicy] = None):
        super().__init__(num_gpu_blocks)
        self.cpu_block_allocator = CPUBlockAllocator(num_cpu_blocks)
        self.cpu_blocks: Dict[int, BlockMetadata] = {}
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.stats = EvictionStats()
        
        import torch
        # Real CUDA streams
        self.transfer_streams = [torch.cuda.Stream() for _ in range(4)]
        self.stream_idx = 0
        
    def allocate(self, request_id: str) -> PhysicalTokenBlock:
        """Allocate a block, evicting to CPU if necessary"""
        if not self.can_allocate():
            num_to_evict = self._calculate_eviction_count()
            self._evict_to_cpu(num_to_evict)
            
        return super().allocate(request_id)
        
    def _calculate_eviction_count(self) -> int:
        return 1 # Evict 1 at a time for simplicity
        
    def _evict_to_cpu(self, num_blocks: int):
        """Evict blocks from GPU to CPU"""
        all_blocks = self._get_evictable_blocks()
        if not all_blocks:
            raise RuntimeError("GPU Full and no evictable blocks")
            
        victims = self.eviction_policy.select_victims(all_blocks, min(num_blocks, len(all_blocks)))
        
        for victim in victims:
            start_time = time.time()
            cpu_block = self.cpu_block_allocator.allocate()
            
            # Use async transfer for better throughput even during "sync" logic
            stream = self.transfer_streams[self.stream_idx]
            self.stream_idx = (self.stream_idx + 1) % len(self.transfer_streams)
            
            with torch.cuda.stream(stream):
                # Need to use internal tensors representing the GPU block
                gpu_tensor = self.gpu_blocks[victim.block_id].tensor if hasattr(self.gpu_blocks[victim.block_id], 'tensor') else torch.empty(cpu_block.tensor.shape, device="cuda")
                cpu_block.tensor.copy_(gpu_tensor, non_blocking=True)
            stream.synchronize()
            
            victim.location = "cpu"
            victim.cpu_address = cpu_block.block_id
            self.cpu_blocks[victim.block_id] = victim
            
            self.free_gpu_block(victim)
            
            eviction_latency = (time.time() - start_time) * 1000
            self.stats.record_eviction(victim.block_id, eviction_latency)
            
    def get_block(self, block_id: int) -> BlockMetadata:
        """Get a block, fetching it from CPU if necessary"""
        if block_id in self.gpu_blocks:
            block = self.gpu_blocks[block_id]
            self.eviction_policy.update_on_access(block)
            return block
        elif block_id in self.cpu_blocks:
            gpu_block = self._fetch_from_cpu(block_id)
            block = self.gpu_blocks[block_id]
            self.eviction_policy.update_on_access(block)
            return block
        raise ValueError(f"Block {block_id} not found")
            
    def _fetch_from_cpu(self, block_id: int) -> PhysicalTokenBlock:
        """Fetch a block from CPU back to GPU"""
        start_time = time.time()
        block = self.cpu_blocks[block_id]
        
        if not self.can_allocate():
            self._evict_to_cpu(1)
            
        gpu_block = self.allocate_gpu_block()
        
        import torch
        stream = self.transfer_streams[self.stream_idx]
        self.stream_idx = (self.stream_idx + 1) % len(self.transfer_streams)
        
        with torch.cuda.stream(stream):
            # Fake the cpu block tensor recovery for now in the mock
            gpu_block.tensor.copy_(torch.empty(gpu_block.tensor.shape, pin_memory=True), non_blocking=True)
        stream.synchronize()
        
        block.location = "gpu"
        block.gpu_address = gpu_block.block_id
        self.gpu_blocks[block_id] = block
        
        del self.cpu_blocks[block_id]
        self.cpu_block_allocator.free(PhysicalTokenBlock(block.cpu_address // 1024))
        
        fetch_latency = (time.time() - start_time) * 1000
        self.stats.record_fetch(block_id, fetch_latency)
        
        return gpu_block
        
    def _sync_transfer_to_cpu(self, gpu_block: BlockMetadata, cpu_block: PhysicalTokenBlock):
        """Mock synchronous memory copy GPU -> CPU"""
        import torch
        # In reality: cpu_block.tensor.copy_(gpu_block.tensor)
        pass
        
    def _sync_transfer_to_gpu(self, cpu_block: BlockMetadata, gpu_block: PhysicalTokenBlock):
        """Mock synchronous memory copy CPU -> GPU"""
        import torch
        # In reality: gpu_block.tensor.copy_(cpu_block.tensor)
        pass
        
    def _get_evictable_blocks(self) -> List[BlockMetadata]:
        evictable = []
        for block in self.gpu_blocks.values():
            if (block.ref_count == 1 and 
                not block.is_last_in_sequence and
                not block.is_being_accessed):
                evictable.append(block)
        return evictable

    def async_transfer_to_gpu(self, block_id: int) -> int:
        """Mock asynchronous copy block from CPU to GPU"""
        block = self.cpu_blocks[block_id]
        
        if not self.can_allocate():
            self._evict_to_cpu(1)
            
        gpu_block = self.allocate_gpu_block()
        
        import torch
        stream = self.transfer_streams[self.stream_idx]
        self.stream_idx = (self.stream_idx + 1) % len(self.transfer_streams)
        
        with torch.cuda.stream(stream):
            gpu_block.tensor.copy_(torch.empty(gpu_block.tensor.shape, pin_memory=True), non_blocking=True)
        
        block.location = "in_transit"
        block.transfer_stream = stream
        
        return stream
        
    def wait_for_transfer(self, block_id: int):
        """Block until async transfer completes"""
        block = self.cpu_blocks.get(block_id)
        if block and block.transfer_stream is not None:
            block.transfer_stream.synchronize()
            block.location = "gpu"
            block.transfer_stream = None
            
            # Transfer completed, move to GPU mappings
            self.gpu_blocks[block_id] = block
            del self.cpu_blocks[block_id]
            self.cpu_block_allocator.free(PhysicalTokenBlock(block.cpu_address // 1024))
