# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    estimate_max_model_len,
    generate_block_hash_extra_keys,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    is_kv_cache_spec_uniform,
    make_block_hash_with_group_id,
    tensor_data,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.metrics.stats import CachingMetrics, PrefixCacheStats
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int] | None,
    block_size: int = 3,
    hash_fn: Callable = hash,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
        prompt_embeds=prompt_embeds,
    )


def new_kv_cache_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=None,
    attention_chunk_size=None,
):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
        attention_chunk_size=attention_chunk_size,
    )


def new_sliding_window_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=1,
):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
    )


def new_chunked_local_attention_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    attention_chunk_size=4,
):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        attention_chunk_size=attention_chunk_size,
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_none_hash(monkeypatch, hash_fn):
    import vllm.v1.core.kv_cache_utils

    # case 1: PYTHONHASHSEED is not set, use random
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert reloaded_kv_cache_utils.NONE_HASH != b""

    # case 2: PYTHONHASHSEED is set, use the seed and hash_fn
    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "python hash seed")
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert hash_fn("python hash seed") == reloaded_kv_cache_utils.NONE_HASH


def test_kv_cache_block():
    # Test KVCacheBlock initialization
    block = KVCacheBlock(block_id=0)
    assert block.block_id == 0
    assert block.ref_cnt == 0
    assert block.block_hash is None

    # Test reference count manipulation
    block.ref_cnt += 1
    assert block.ref_cnt == 1
    block.ref_cnt -= 1
    assert block.ref_cnt == 0

    # Test block hash setting and resetting
    block_hash = make_block_hash_with_group_id(BlockHash(b"abc"), 0)
    block.block_hash = block_hash
    assert block.block_hash == block_hash

    block.reset_hash()
    assert block.block_hash is None


def test_free_kv_cache_block_queue_initialization():
    # Test with a single block
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([block])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is block
    assert queue.fake_free_list_tail.prev_free_block is block


def test_free_kv_cache_block_queue_operations():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check initial state
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Pop the first block
    block1 = queue.popleft()
    assert block1 == blocks[0]
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Remove a block from the middle
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.num_free_blocks == 3
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]

    # Append a block back
    queue.append(block_to_remove)
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_tail.prev_free_block is block_to_remove
    assert block_to_remove.prev_free_block is blocks[4]
    assert block_to_remove.next_free_block is queue.fake_free_list_tail

    # Pop blocks until empty
    for _ in range(4):
        queue.popleft()
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head

    # Attempt to pop from an empty queue
    with pytest.raises(ValueError) as e:
        queue.popleft()
    assert str(e.value) == "No free blocks available"


def test_free_kv_cache_block_queue_append_n():
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue([])
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Append 0 block
    # fake_head->fake_tail
    queue.append_n([])
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head
    # Append 1 block
    # fake_head->b0->fake_tail
    queue.append_n(blocks[0:1])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]
    # Append 2 blocks
    # fake_head->b0->b4->b5->fake_tail
    queue.append_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[5]
    # Append 3 blocks
    # fake_head->b0->b4->b5->b1->b2->b3->fake_tail
    queue.append_n(blocks[1:4])
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[1]
    assert blocks[1].prev_free_block is blocks[5]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[1]
    assert blocks[2].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[2]
    assert blocks[3].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[3]

    # Create an empty FreeKVCacheBlockQueue
    invalid_queue = FreeKVCacheBlockQueue([])
    # set prev_free_block to None and this will cause assertation in append_n
    invalid_queue.fake_free_list_tail.prev_free_block = None
    with pytest.raises(AssertionError):
        # Append 1 block
        # fake_head->fake_tail
        invalid_queue.append_n(blocks[0:1])
    assert invalid_queue.num_free_blocks == 0
    assert (
        invalid_queue.fake_free_list_head.next_free_block
        == invalid_queue.fake_free_list_tail
    )


def test_free_kv_cache_block_queue_popleft_n():
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(
        [blocks[1], blocks[3], blocks[5], blocks[4], blocks[0], blocks[2]]
    )
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].prev_free_block is queue.fake_free_list_head
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]
    assert blocks[3].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[3]
    assert blocks[5].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[5]
    assert blocks[4].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[4]
    assert blocks[0].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[0]
    assert blocks[2].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[2]

    # Pop 0 block
    # fake_head->b1->b3->b5->b4->b0->b2->fake_tail
    assert len(queue.popleft_n(0)) == 0
    assert queue.num_free_blocks == 6
    # Pop 1 block
    # fake_head->b3->b5->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(1)
    assert queue.num_free_blocks == 5
    assert len(result_blocks) == 1
    assert result_blocks[0] is blocks[1]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 2 blocks
    # fake_head->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(2)
    assert len(result_blocks) == 2
    assert queue.num_free_blocks == 3
    assert result_blocks[0] is blocks[3]
    assert result_blocks[1] is blocks[5]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 3 blocks
    # fake_head->fake_tail
    result_blocks = queue.popleft_n(3)
    assert len(result_blocks) == 3
    assert queue.num_free_blocks == 0
    assert result_blocks[0] is blocks[4]
    assert result_blocks[1] is blocks[0]
    assert result_blocks[2] is blocks[2]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


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
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:] + [block_to_remove]


def test_generate_block_hash_extra_keys():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=10, length=5),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys == ("hash1",)
    assert next_mm_idx == 1

    # Test with partial overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 3, 8, 0)
    assert extra_keys == ("hash1",)
    assert next_mm_idx == 1

    # Test with no overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None
    assert next_mm_idx == 1

    # Test with multiple extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 15, 0)
    assert extra_keys == ("hash1", "hash2")
    assert next_mm_idx == 2


def test_generate_block_hash_extra_keys_no_mm_inputs():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys is None
    assert next_mm_idx == 0


def test_generate_block_hash_extra_keys_cache_salt():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
        cache_salt="salt",
    )

    # salt is added for the first token
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 1, 0)
    assert extra_keys == ("salt",)
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 10, 0)
    assert extra_keys == ("salt",)

    # no salt added for other tokens
    extra_keys, _ = generate_block_hash_extra_keys(request, 1, 2, 0)
    assert extra_keys is None
    extra_keys, _ = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None

    # works together with other extra keys
    request_mm = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
        ],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request_mm, 0, 5, 0)
    assert extra_keys == ("hash1", "salt")
    assert next_mm_idx == 1


def test_generate_block_hash_extra_keys_prompt_embeds():
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
    )

    # Test with prompt embeds for the first block
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    expected_embeds = prompt_embeds[0:5]
    expected_bytes = kv_cache_utils.tensor_data(expected_embeds).tobytes()
    assert extra_keys == (expected_bytes,)

    # Test with prompt embeds for the second block
    extra_keys, _ = generate_block_hash_extra_keys(request, 5, 10, 0)
    expected_embeds = prompt_embeds[5:10]
    expected_bytes = kv_cache_utils.tensor_data(expected_embeds).tobytes()
    assert extra_keys == (expected_bytes,)


def test_generate_block_hash_extra_keys_different_prompt_embeds():
    prompt_embeds1 = torch.randn(10, 3)
    prompt_embeds2 = torch.randn(10, 3)
    request1 = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds1,
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds2,
    )

    extra_keys1, _ = generate_block_hash_extra_keys(request1, 0, 5, 0)
    extra_keys2, _ = generate_block_hash_extra_keys(request2, 0, 5, 0)
    assert extra_keys1 != extra_keys2


def test_generate_block_hash_extra_keys_lora():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
    )

    request.lora_request = LoRARequest(
        lora_name="test_lora_adapter", lora_int_id=1, lora_path="/path/to/lora"
    )

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys == ("test_lora_adapter",)

    request.lora_request = None
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys is None


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_block_tokens(hash_fn):
    parent_block_hash = BlockHash(b"123")
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(
        hash_fn, parent_block_hash, curr_block_token_ids, extra_keys
    )
    expected = hash_fn((parent_block_hash, curr_block_token_ids, extra_keys))
    assert block_hash == expected


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn((kv_cache_utils.NONE_HASH, (0, 1, 2), ("hash1",)))
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), ("hash2",)))


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_tokens_different_mm_input(hash_fn):
    request1 = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash3", "hash2"],
    )
    block_hashes1 = request1.block_hashes
    block_hashes2 = request2.block_hashes
    assert block_hashes1[0] != block_hashes2[0]
    assert block_hashes1[1] != block_hashes2[1]


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_request_tokens_no_mm_inputs(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=None,
        mm_hashes=None,
    )

    block_hashes = request.block_hashes

    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn((kv_cache_utils.NONE_HASH, (0, 1, 2), None))
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), None))


def _stats(requests: int, queries: int, hits: int) -> PrefixCacheStats:
    return PrefixCacheStats(requests=requests, queries=queries, hits=hits)


def test_metrics():
    """
    Test the prefix caching metrics.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    assert metrics.hit_rate == 0.0

    metrics.observe(_stats(1, 20, 9))
    # 9 / 20 = 0.45
    assert metrics.hit_rate == 0.45

    metrics.observe(_stats(4, 80, 16))

    # 25 / 100 = 0.25
    assert metrics.hit_rate == 0.25

    metrics.observe(_stats(1, 10, 2))

    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    metrics.reset()
    assert metrics.hit_rate == 0.0
    assert metrics.aggregated_requests == 0
    assert metrics.aggregated_query_total == 0
    assert metrics.aggregated_query_hit == 0
    assert not metrics.query_queue


def test_metrics_empty_stats():
    """
    Test the prefix caching metrics with empty stats.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 20, 9))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(4, 80, 16))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 10, 2))
    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    # Only the latest added stats preserved 10 / 20 = 0.5
    metrics.observe(_stats(11, 20, 10))
    assert metrics.aggregated_requests == 11
    assert metrics.aggregated_query_total == 20
    assert metrics.aggregated_query_hit == 10
    assert metrics.hit_rate == 0.5

    # Only the latest added stats preserved 30 / 40 = 0.75
    metrics.observe(_stats(22, 40, 30))
    assert metrics.aggregated_requests == 22
    assert metrics.aggregated_query_total == 40
    assert metrics.aggregated_query_hit == 30
    assert metrics.hit_rate == 0.75


def test_get_kv_cache_configs_multiple_workers():
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    same_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
    ]

    # Basic case. All things are the same.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different available memory. This is the case for TP.
    # Use the smallest memory available.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 20,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different KV cache specs. This is the case for PP.
    different_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer2": new_kv_cache_spec(),
            "layer3": new_kv_cache_spec(),
        },
    ]

    # Different workers have different layers.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2", "layer3"], new_kv_cache_spec()),
            ],
        ),
    ]

    # Some layers are the same, some are different. This is the case for TP+PP
    tp_pp_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
    ]

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        tp_pp_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different workers have different types of layers. This is the case for
    # hybrid models + PP.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_sliding_window_spec(),
            "layer4": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
                KVCacheGroupSpec([], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer4"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec([], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer3", "layer4"], new_sliding_window_spec()),
            ],
        ),
    ]

    # When divided into multiple KVCacheGroups, need to ensure the number of
    # layers per group is similar.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_sliding_window_spec(),
            "layer3": new_sliding_window_spec(),
        },
        {
            "layer4": new_kv_cache_spec(),
            "layer5": new_sliding_window_spec(),
            "layer6": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 10,
            ref_kv_cache_spec.page_size_bytes * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer1", "layer2", "layer3"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer2"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer3"], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer4", "layer5", "layer6"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer4"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer5"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer6"], new_sliding_window_spec()),
            ],
        ),
    ]

    # Have conflicting layers. Need to raise an error.
    conflicting_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer1": new_sliding_window_spec(),
        },
    ]
    with pytest.raises(AssertionError):
        get_kv_cache_configs(
            vllm_config,
            conflicting_layer_specs,
            [
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ],
        )


def test_merge_kv_cache_spec():
    same_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32),
    ]
    merged_layer_spec = same_layer_specs[0].merge(same_layer_specs)
    assert merged_layer_spec.block_size == 16
    assert merged_layer_spec.num_kv_heads == 32
    assert merged_layer_spec.head_size == 64
    assert merged_layer_spec.dtype == torch.float32
    assert merged_layer_spec.sliding_window is None

    different_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=16),
    ]
    with pytest.raises(AssertionError):
        different_layer_specs[0].merge(different_layer_specs)

    full_spec = new_kv_cache_spec(num_kv_heads=32)
    different_type_layer_specs = [
        full_spec,
        SlidingWindowSpec(
            block_size=full_spec.block_size,
            num_kv_heads=full_spec.num_kv_heads,
            head_size=full_spec.head_size,
            dtype=full_spec.dtype,
            sliding_window=1,
        ),
    ]
    with pytest.raises(AssertionError):
        different_type_layer_specs[0].merge(different_type_layer_specs)
    with pytest.raises(AssertionError):
        different_type_layer_specs[1].merge(different_type_layer_specs)

    different_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=2),
    ]
    with pytest.raises(ValueError):
        different_sliding_window_layer_specs[0].merge(
            different_sliding_window_layer_specs
        )

    same_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    ]
    merged_layer_spec = same_sliding_window_layer_specs[0].merge(
        same_sliding_window_layer_specs
    )
    assert merged_layer_spec.sliding_window == 1

    same_sliding_window_layer_spec_with_none = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=None),
    ]
    merged_layer_spec = same_sliding_window_layer_spec_with_none[0].merge(
        same_sliding_window_layer_spec_with_none
    )
    assert merged_layer_spec.sliding_window == 1


def test_is_kv_cache_spec_uniform():
    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=2),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"),
    [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ],
)
def test_estimate_max_model_len(model_id, max_model_len, want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    # Create KV cache specs
    kv_cache_spec = {}
    for i in range(32):
        layer_name = f"layer_{i}"
        kv_cache_spec[layer_name] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            dtype=torch.float16,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(
        vllm_config, kv_cache_spec, 8 * GiB_bytes
    )
    assert estimated_max_len == want_estimated_max_len


def test_get_max_concurrency_for_kv_cache_config():
    # Create a VllmConfig
    model_id = "Qwen/Qwen1.5-7B"
    max_model_len = 16384
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
    )

    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        sliding_window=1024,
    )

    kv_cache_config_full_attention = KVCacheConfig(
        num_blocks=int(1024 * 1.5),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
        ],
    )
    max_concurrency_full_attention = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_full_attention
    )
    assert max_concurrency_full_attention == 1.5

    kv_cache_config_sliding_window = KVCacheConfig(
        num_blocks=129 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], sliding_window_spec),
        ],
    )
    max_concurrency_sliding_window = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_sliding_window
    )
    assert max_concurrency_sliding_window == 3

    kv_cache_config_hybrid_model = KVCacheConfig(
        num_blocks=(1024 + 129) * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
            KVCacheGroupSpec(
                [f"layer_{i}" for i in range(32, 64)], sliding_window_spec
            ),
        ],
    )
    max_concurrency_hybrid_model = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert max_concurrency_hybrid_model == 3


def test_allocate_with_lookahead():
    """Verify that lookahead tokens correctly affect block allocation"""
    block_size = 4
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=100, shared_by=["layer1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], new_kv_cache_spec(block_size=block_size)),
        ],
    )

    request = make_request(
        request_id="0",
        prompt_token_ids=[],
        block_size=block_size,
        mm_positions=None,
        mm_hashes=None,
    )

    # Test case 1: Requires additional lookahead tokens
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,  # Total required: 3+2=5 tokens
    )
    assert len(blocks.get_block_ids()[0]) == 2  # ceil(5/4)=2 blocks

    # Test case 2: With precomputed blocks
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    # required_blocks = ceil((3 + 2) /4) = 2
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,
    )
    assert len(blocks.get_block_ids()[0]) == 2

    # Test case 3: With precomputed blocks
    # required_blocks = ceil((3 + 4) / 4) = 2
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config, max_model_len=100, hash_block_size=block_size
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=4,
    )
    assert len(blocks.get_block_ids()[0]) == 2


def test_get_kv_cache_config_one_worker():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    print(kv_cache_config_full)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        "layer_1": new_sliding_window_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_sliding], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ],
    )

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"], new_kv_cache_spec(sliding_window=1)
            ),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 64, shared_by=["layer_1", "layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_sliding_window_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_3", "layer_4"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_5", "layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_5"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_4", "layer_6"], new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_4", "layer_5", "layer_6"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_7", "layer_8", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32, shared_by=["layer_3", "layer_10"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"], new_kv_cache_spec()),
            KVCacheGroupSpec(
                ["layer_4", "layer_7", "layer_10"], new_sliding_window_spec()
            ),
            KVCacheGroupSpec(["layer_5", "layer_8"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_6", "layer_9"], new_sliding_window_spec()),
        ],
    )

    # 6 full + 5 sliding, pad to 6 full + 6 sliding. This is a typical case for gpt-oss
    # eagle where there is only one more full attention layer than sliding window layers
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_kv_cache_spec(),
        "layer_5": new_kv_cache_spec(),
        "layer_6": new_kv_cache_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
        "layer_11": new_sliding_window_spec(),
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 6 * 32]
    )[0]
    print(kv_cache_config_hybrid)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_7"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_8"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_3", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_4", "layer_10"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_5", "layer_11"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                new_kv_cache_spec(),
            ),
            KVCacheGroupSpec(
                ["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                new_sliding_window_spec(),
            ),
        ],
    )

    # different hidden size but same type, use UniformTypeKVCacheSpecs
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=128),
        "layer_2": new_kv_cache_spec(head_size=64),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32 * 2, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"],
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs=kv_cache_specs_hybrid
                ),
            )
        ],
    )

    # Different hidden size and different type, align by different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=32),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32, shared_by=["layer_1", "layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(
                ["layer_2"], new_sliding_window_spec(head_size=32, block_size=32)
            ),
        ],
    )

    # different hidden size that cannot be aligned by using different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=96),
    }

    with pytest.raises(NotImplementedError):
        get_kv_cache_configs(
            vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
        )[0]

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    kv_cache_configs = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )
    ]


def test_generate_uniform_type_kv_cache_specs():
    # All layers are full attention, can be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # Full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(sliding_window=1),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # different order of full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_kv_cache_spec(),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # Same-size sliding window, can be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_sliding_window_spec(sliding_window=1, head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # different block sizes, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(block_size=16),
        "layer_2": new_kv_cache_spec(block_size=32),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None


def test_generate_scheduler_kv_cache_config():
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    kv_cache_configs = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer_1", "layer_2"],
                    UniformTypeKVCacheSpecs(
                        block_size=16, kv_cache_specs=kv_cache_specs
                    ),
                ),
            ],
        )
    ]
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    assert scheduler_kv_cache_config == KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def new_mla_spec(cache_dtype_str=None):
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=16,
        head_size=64,
        dtype=torch.float32,
        cache_dtype_str=cache_dtype_str,
    )


def test_merge_mla_spec():
    kv_cache_specs = [
        new_mla_spec(),
        new_mla_spec(),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec()

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec(cache_dtype_str="fp8_ds_mla")

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str=None),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_kv_cache_spec(),
        new_mla_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_kv_cache_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher_with_prompt_embeds(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_bytes = tensor_data(prompt_embeds[:block_size]).tobytes()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (block1_embeds_bytes,),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_bytes = tensor_data(prompt_embeds[block_size:num_tokens]).tobytes()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (block2_embeds_bytes,),
        )
    )
    assert block_hashes[1] == expected_hash2


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_with_prompt_embeds_and_mm_inputs(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_bytes = tensor_data(prompt_embeds[:block_size]).tobytes()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            ("hash1", block1_embeds_bytes),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_bytes = tensor_data(prompt_embeds[block_size:num_tokens]).tobytes()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            ("hash2", block2_embeds_bytes),
        )
    )
    assert block_hashes[1] == expected_hash2


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=1024)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # With enough memory, max_model_len stays at the derived max
    large_available_memory = mem_per_block_per_layer * 2 * 1024  # plenty of memory
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [large_available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024

    # Reset for next test
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    # With limited memory, max_model_len should be reduced
    # Need memory for at least max_model_len tokens
    # 32 blocks worth of memory for 2 layers = can fit 32*16=512 tokens
    limited_memory = mem_per_block_per_layer * 2 * 32
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [limited_memory]
    )
    # Should be reduced to fit in memory
    assert vllm_config.model_config.max_model_len < 1024
    assert vllm_config.model_config.max_model_len > 0


def test_auto_fit_max_model_len_not_triggered():
    """Test that auto-fit is not triggered when original_max_model_len is not -1."""
    model_config = ModelConfig(max_model_len=16)
    # original_max_model_len should be None by default, not -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # This should work normally without auto-fit
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_per_block_per_layer * 2 * 32]
    )
    assert vllm_config.model_config.max_model_len == 16


def test_unify_hybrid_kv_cache_specs():
    # 1. has_full_attention and has_sliding_window
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 2. has_full_attention and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )

    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 3. has_full_attention, has_sliding_window and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    before_spec_3 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
        "layer_3": before_spec_3,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    expected_spec_3 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_3"] == expected_spec_3

    # 4. No FullAttentionSpec, should not convert
    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(sliding_window=1024),
        "layer_2": new_chunked_local_attention_spec(attention_chunk_size=512),
    }

    with pytest.raises(ValueError):
        kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
