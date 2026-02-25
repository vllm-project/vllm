from vllm_extensions.eviction_policies import LRUEvictionPolicy, BlockMetadata
import time

def test_lru_policy():
    policy = LRUEvictionPolicy()
    
    blocks = [
        BlockMetadata(block_id=0, last_access_time=time.time() - 10),
        BlockMetadata(block_id=1, last_access_time=time.time() - 5),
        BlockMetadata(block_id=2, last_access_time=time.time() - 1),
    ]
    
    victims = policy.select_victims(blocks, num_to_evict=1)
    
    assert len(victims) == 1
    assert victims[0].block_id == 0  # Oldest block
    print("✓ LRU test passed!")

if __name__ == "__main__":
    test_lru_policy()
