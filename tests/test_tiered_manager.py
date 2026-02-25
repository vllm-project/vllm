import pytest
from vllm_extensions.eviction_policies import LRUEvictionPolicy
from vllm_extensions.tiered_block_manager import TieredBlockSpaceManager

def test_basic_eviction():
    """Test that eviction happens when GPU full"""
    manager = TieredBlockSpaceManager(
        num_gpu_blocks=10,
        num_cpu_blocks=100,
        eviction_policy=LRUEvictionPolicy()
    )
    
    # Fill GPU
    blocks = [manager.allocate(f"req_{i}") for i in range(10)]
    assert len(manager.gpu_blocks) == 10
    assert len(manager.cpu_blocks) == 0
    
    # Trigger eviction
    manager.allocate("req_11")
    assert len(manager.cpu_blocks) >= 1
    assert len(manager.gpu_blocks) == 10

def test_cpu_fetch():
    """Test fetching evicted block back to GPU"""
    manager = TieredBlockSpaceManager(
        num_gpu_blocks=2,
        num_cpu_blocks=10,
        eviction_policy=LRUEvictionPolicy()
    )
    
    # Fill GPU
    b0 = manager.allocate("req_0")
    b1 = manager.allocate("req_1")
    
    # Trigger eviction of b0
    b2 = manager.allocate("req_2")
    assert b0.block_id in manager.cpu_blocks
    assert b0.block_id not in manager.gpu_blocks
    
    # Fetch b0 back (might evict b1)
    block = manager.get_block(b0.block_id)
    
    assert block.location == "gpu"
    assert b0.block_id not in manager.cpu_blocks

def test_shared_block_not_evicted():
    """Test that shared blocks (ref_count > 1) are not evicted"""
    manager = TieredBlockSpaceManager(num_gpu_blocks=5, num_cpu_blocks=10)
    
    # Create shared block and fill GPU
    shared_block = manager.allocate("req_1")
    # Need to manually get internal representation to adjust ref_count for mock
    manager.gpu_blocks[shared_block.block_id].ref_count = 2
    
    for i in range(4):
        manager.allocate(f"req_{i+2}")
        
    # Trigger eviction
    manager.allocate("req_6")
    
    # Shared block should still be on GPU
    assert shared_block.block_id not in manager.cpu_blocks
