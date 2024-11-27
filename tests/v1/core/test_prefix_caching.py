"""Compare the with and without prefix caching."""
from vllm.inputs import token_inputs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import hash_block_tokens


def make_request(request_id, prompt_token_ids):
    return Request(
        request_id=request_id,
        inputs=token_inputs(prompt_token_ids=prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=17),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
    )


def test_prefill():
    manager = KVCacheManager(
        block_size=16,
        num_gpu_blocks=10,
        sliding_window=False,
        enable_caching=True,
        num_preallocate_tokens=16,
    )

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    req0 = make_request("0", common_token_ids + unique_token_ids)
    computed_blocks = manager.get_computed_blocks(req0)
    assert not computed_blocks
    blocks = manager.allocate_slots(req0, 55, computed_blocks)
    assert [b.block_id for b in blocks] == [0, 1, 2, 3, 4]

    # Check full block metadata
    parent_block_hash = None
    for block_id in (0, 1, 2):
        block_hash = hash_block_tokens(parent_block_hash,
                                       manager.block_pool[block_id].token_ids)
        assert manager.block_pool[block_id].block_hash == block_hash
        assert manager.block_pool[block_id].ref_cnt == 1
        assert manager.block_pool[block_id].num_hashed_tokens == 16 * (
            block_id + 1)
        assert manager.block_pool[block_id].token_ids == tuple([block_id] * 16)
        parent_block_hash = block_hash

    # Check partial/preallocated block metadata
    for block_id in (3, 4):
        assert manager.block_pool[block_id].block_hash is None
        assert manager.block_pool[block_id].ref_cnt == 1
        assert manager.block_pool[block_id].num_hashed_tokens == 0
        if block_id == 3:
            assert manager.block_pool[block_id].token_ids == [3] * 7
        else:
            assert not manager.block_pool[block_id].token_ids

    # Cache hit in the common prefix when the original block is still in use.
    # Incomplete 1 block (5 tokens)
    unique_token_ids = [3] * 5
    req1 = make_request("1", common_token_ids + unique_token_ids)
    computed_blocks = manager.get_computed_blocks(req1)
    assert [b.block_id for b in computed_blocks] == [0, 1, 2]
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, computed_blocks)
    assert [b.block_id for b in blocks] == [5, 6]
    for block in computed_blocks:
        assert block.ref_cnt == 2

    # At this point, we should have 3 free blocks left.
    assert manager.free_block_queue.num_free_blocks == 3

    manager.free(req0)
    manager.free(req1)

    # All blocks should be available.
    assert manager.free_block_queue.num_free_blocks == 10
    # The order should be
    # [unallocated (7, 8)]
    # [unique_req0 (4, 3)]
    # [unique_req1 (6, 5)]
    # [common (2, 1, 0)]
    assert [
        b.block_id for b in manager.free_block_queue.get_all_free_blocks()
    ] == [7, 8, 9, 4, 3, 6, 5, 2, 1, 0]

    # Cache hit in the common prefix when the original block is already free.
    # Incomplete 1 block (6 tokens)
    unique_token_ids = [3] * 6
    req2 = make_request("2", common_token_ids + unique_token_ids)
    computed_block = manager.get_computed_blocks(req2)
    assert [b.block_id for b in computed_block] == [0, 1, 2]
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req2, num_new_tokens, computed_blocks)
    assert [b.block_id for b in blocks] == [7, 8]

    # Although we only have 5 free blocks, we have 8 blocks in
    # the free block queue due to lazy removal.
    assert manager.free_block_queue.num_free_blocks == 5
    assert all([
        b.ref_cnt == 0 for b in manager.free_block_queue.get_all_free_blocks()
    ])
    assert len([b
                for b in manager.free_block_queue.get_all_free_blocks()]) == 5

    manager.free(req2)

    # Cache miss and eviction.
    req3 = make_request("3", [99] * (16 * 9))
    computed_blocks = manager.get_computed_blocks(req3)
    assert not computed_blocks
    blocks = manager.allocate_slots(req2, 16 * 9, computed_blocks)
    # This block ID order also checks the eviction order.
    assert [b.block_id for b in blocks] == [9, 4, 3, 6, 5, 8, 7, 2, 1, 0]
    assert manager.free_block_queue.num_free_blocks == 0
    assert manager.free_block_queue.free_list_head is None
    assert manager.free_block_queue.free_list_tail is None


def test_decode():
    manager = KVCacheManager(
        block_size=16,
        num_gpu_blocks=10,
        sliding_window=False,
        enable_caching=True,
        num_preallocate_tokens=16,
    )

    # Complete 3 blocks (48 tokens)
    common_token_ids = [i for i in range(3) for _ in range(16)]

    # Fully cache miss
    # Incomplete 1 block (7 tokens)
    unique_token_ids = [3] * 7
    req0 = make_request("0", common_token_ids + unique_token_ids)
    computed_blocks = manager.get_computed_blocks(req0)
    assert not computed_blocks
    blocks = manager.allocate_slots(req0, 55, computed_blocks)
    assert [b.block_id for b in blocks] == [0, 1, 2, 3, 4]

    # Append slots without allocating a new block.
    req0.num_computed_tokens = 55
    for _ in range(4):
        req0.append_output_token_ids(8)
    new_blocks = manager.append_slots(req0, 4)
    assert new_blocks is not None and len(new_blocks) == 0
    assert len(manager.block_pool[3].token_ids) == 11

    # Append slots without allocating a new block, but start using the
    # preallocated block.
    req0.num_computed_tokens = 59
    # 6 tokens to fill the previous block, and 10 tokens to fill
    # the preallocated block.
    for _ in range(5 + 10):
        req0.append_output_token_ids(7)
    new_blocks = manager.append_slots(req0, 15)
    assert new_blocks is not None and len(new_blocks) == 0
    assert len(manager.block_pool[3].token_ids) == 16
    assert len(manager.block_pool[4].token_ids) == 10

    # Append slots with allocating a new block.
    req0.num_computed_tokens = 74
    # 6 tokens to fill the previous block, and 10 tokens to fill
    # the preallocated block.
    for _ in range(6 + 11):
        req0.append_output_token_ids(12)
    new_blocks = manager.append_slots(req0, 17)
    # Plus one preallocated block.
    assert new_blocks is not None and len(new_blocks) == 2
    assert len(manager.block_pool[4].token_ids) == 16
    assert len(manager.block_pool[5].token_ids) == 11
    assert len(manager.block_pool[6].token_ids) == 0


def test_evict():
    manager = KVCacheManager(
        block_size=16,
        num_gpu_blocks=10,
        sliding_window=False,
        enable_caching=True,
        num_preallocate_tokens=16,
    )

    last_token_id = 5 * 16 + 7
    req0 = make_request("0", list(range(last_token_id)))
    computed_blocks = manager.get_computed_blocks(req0)
    assert not computed_blocks
    blocks = manager.allocate_slots(req0, 5 * 16 + 7, computed_blocks)
    assert len(blocks) == 7  # 5 full + 1 partial + 1 preallocated

    # 3 blocks.
    req1 = make_request("1", list(range(last_token_id,
                                        last_token_id + 3 * 16)))
    computed_blocks = manager.get_computed_blocks(req1)
    assert not computed_blocks
    blocks = manager.allocate_slots(req1, 3 * 16, computed_blocks)
    assert len(blocks) == 3  # 3 full blocks
    last_token_id += 3 * 16

    assert manager.free_block_queue.num_free_blocks == 0

    manager.free(req0)
    manager.free(req1)
    assert manager.free_block_queue.num_free_blocks == 10
    assert [
        b.block_id for b in manager.free_block_queue.get_all_free_blocks()
    ] == [6, 5, 4, 3, 2, 1, 0, 9, 8, 7]

    # Touch the first 2 blocks.
    req2 = make_request("2", list(range(2 * 16 + 3)))
    computed_blocks = manager.get_computed_blocks(req2)
    assert [b.block_id for b in computed_blocks] == [0, 1]
    blocks = manager.allocate_slots(req2, 3, computed_blocks)
    assert [b.block_id for b in blocks] == [6, 5]
    assert manager.free_block_queue.num_free_blocks == 6
