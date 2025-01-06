import pytest

from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens)
from vllm.v1.request import Request


def make_request(request_id,
                 prompt_token_ids,
                 mm_positions=None,
                 mm_hashes=None):
    if mm_positions is None:
        multi_modal_inputs = None
    else:
        multi_modal_inputs = [MultiModalKwargs({})] * len(mm_positions)

    return Request(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        multi_modal_inputs=multi_modal_inputs,
        multi_modal_hashes=mm_hashes,
        multi_modal_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
    )


def test_kv_cache_block():
    # Test KVCacheBlock initialization
    block = KVCacheBlock(block_id=0)
    assert block.block_id == 0
    assert block.ref_cnt == 0
    assert block.block_hash is None

    # Test reference count manipulation
    block.incr_ref()
    assert block.ref_cnt == 1
    block.decr_ref()
    assert block.ref_cnt == 0

    # Test block hash setting and resetting
    block_hash = BlockHashType(hash_value=123, token_ids=(1, 2, 3))
    block.block_hash = block_hash
    assert block.block_hash == block_hash

    block.reset_hash()
    assert block.block_hash is None


def test_free_kv_cache_block_queue_initialization():
    # Test with a single block
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([block])
    assert queue.num_free_blocks == 1
    assert queue.free_list_head == block
    assert queue.free_list_tail == block


def test_free_kv_cache_block_queue_operations():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check initial state
    assert queue.num_free_blocks == 5
    assert queue.free_list_head == blocks[0]
    assert queue.free_list_tail == blocks[4]

    # Pop the first block
    block1 = queue.popleft()
    assert block1 == blocks[0]
    assert queue.num_free_blocks == 4
    assert queue.free_list_head == blocks[1]
    assert queue.free_list_tail == blocks[4]

    # Remove a block from the middle
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.num_free_blocks == 3
    assert blocks[1].next_free_block == blocks[3]
    assert blocks[3].prev_free_block == blocks[1]

    # Append a block back
    queue.append(block_to_remove)
    assert queue.num_free_blocks == 4
    assert queue.free_list_tail == block_to_remove
    assert block_to_remove.prev_free_block == blocks[4]
    assert block_to_remove.next_free_block is None

    # Pop blocks until empty
    for _ in range(4):
        queue.popleft()
    assert queue.num_free_blocks == 0
    assert queue.free_list_head is None
    assert queue.free_list_tail is None

    # Attempt to pop from an empty queue
    with pytest.raises(ValueError) as e:
        queue.popleft()
    assert str(e.value) == "No free blocks available"


def test_free_kv_cache_block_queue_get_all_free_blocks():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check all blocks are correctly retrieved
    assert queue.get_all_free_blocks() == blocks

    # Pop a block and check again
    queue.popleft()
    assert queue.get_all_free_blocks() == blocks[1:]

    # Remove a block and check again
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:]

    # Append a block back and check again
    queue.append(block_to_remove)
    assert queue.get_all_free_blocks() == \
        blocks[1:2] + blocks[3:] + [block_to_remove]


def test_generate_block_hash_extra_keys():
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[{
            "offset": 0,
            "length": 5
        }, {
            "offset": 10,
            "length": 5
        }],
        mm_hashes=["hash1", "hash2"],
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys == ("hash1", )
    assert next_mm_idx == 1

    # Test with partial overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 3, 8, 0)
    assert extra_keys == ("hash1", )
    assert next_mm_idx == 1

    # Test with no overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys == ()
    assert next_mm_idx == 1

    # Test with multiple extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 15, 0)
    assert extra_keys == ('hash1', 'hash2')
    assert next_mm_idx == 2


def test_generate_block_hash_extra_keys_no_mm_inputs():
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys is None
    assert next_mm_idx == 0


def test_hash_block_tokens():
    parent_block_hash = 123
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(parent_block_hash, curr_block_token_ids,
                                   extra_keys)
    assert isinstance(block_hash, BlockHashType)
    assert block_hash.hash_value == hash(
        (parent_block_hash, *curr_block_token_ids))
    assert block_hash.token_ids == curr_block_token_ids
    assert block_hash.extra_keys == extra_keys


def test_hash_request_tokens():
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[{
            "offset": 0,
            "length": 3
        }, {
            "offset": 3,
            "length": 3
        }],
        mm_hashes=["hash1", "hash2"],
    )

    block_size = 3
    block_hashes = hash_request_tokens(block_size, request)

    assert len(block_hashes) == 2
    assert isinstance(block_hashes[0], BlockHashType)
    assert isinstance(block_hashes[1], BlockHashType)

    # Check the first block
    assert block_hashes[0].token_ids == (0, 1, 2)
    assert block_hashes[0].extra_keys == ("hash1", )

    # Check the second block
    assert block_hashes[1].token_ids == (3, 4, 5)
    assert block_hashes[1].extra_keys == ("hash2", )


def test_hash_request_tokens_no_mm_inputs():
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    block_size = 3
    block_hashes = hash_request_tokens(block_size, request)

    assert len(block_hashes) == 2
    assert block_hashes[0].token_ids == (0, 1, 2)
    assert block_hashes[0].extra_keys is None
    assert block_hashes[1].token_ids == (3, 4, 5)
    assert block_hashes[1].extra_keys is None
