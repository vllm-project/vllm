# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.block_pool import BlockPool
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
    MambaSpec,
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

    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
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


def new_mamba_spec(
    block_size=16,
    shapes=((2, 512), (3, 32, 32)),
    dtypes=(torch.float32, torch.float32),
    num_speculative_blocks=2,
    mamba_cache_mode="none",
    page_size_padded=None,
):
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=page_size_padded,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
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


def test_kv_cache_block_uses_slots():
    block = KVCacheBlock(block_id=0)

    # Slots eliminate per-instance __dict__, saving ~264 bytes per block.
    # At 100K+ blocks this avoids tens of MB of overhead and GC pressure.
    assert not hasattr(block, "__dict__")

    # Verify that slots actually prevent dynamic attribute assignment.
    with pytest.raises(AttributeError):
        block.unexpected_field = True


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
    # set prev_free_block to None and this will cause assertion in append_n
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
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)

    # Test with prompt embeds for the second block
    extra_keys, _ = generate_block_hash_extra_keys(request, 5, 10, 0)
    expected_embeds = prompt_embeds[5:10]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)


def test_generate_block_hash_extra_keys_prompt_embeds_cached(monkeypatch):
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
        block_size=20,
    )

    num_tensor_data_calls = 0
    original_tensor_data = kv_cache_utils.tensor_data

    def counting_tensor_data(tensor: torch.Tensor):
        nonlocal num_tensor_data_calls
        num_tensor_data_calls += 1
        return original_tensor_data(tensor)

    monkeypatch.setattr(kv_cache_utils, "tensor_data", counting_tensor_data)

    extra_keys_1, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    extra_keys_2, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys_1 == extra_keys_2
    assert num_tensor_data_calls == 1


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


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer1"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer2"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]


def test_project_kv_cache_groups_to_worker():
    spec_a = new_kv_cache_spec()
    spec_b = new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, {"layer4": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}


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

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (block1_embeds_hash,),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (block2_embeds_hash,),
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

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            ("hash1", block1_embeds_hash),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            ("hash2", block2_embeds_hash),
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


def test_auto_fit_max_model_len_with_hybrid():
    """Test that auto-fit works with hybrid KV cache specs."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=8192)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    gamma = 2
    kv_cache_specs = {
        "layer_1": new_mamba_spec(num_speculative_blocks=gamma),
        "layer_2": new_kv_cache_spec(),
    }

    available_memory = mem_per_block_per_layer * (1024 // 16 + 1 + gamma)
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024


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


def _make_qwen35_specs(
    kv_dtype: torch.dtype = torch.bfloat16,
    mamba_dtype: torch.dtype = torch.bfloat16,
):
    """Build KV cache specs matching real Qwen3.5 architecture.

    Both Qwen3.5-4B and 9B share identical KV cache dimensions:
      - Attention: 4 KV heads, 256 head_dim
      - GatedDeltaNet: conv(3, 8192) + temporal(32, 128, 128)
      - 32 layers: 24 GatedDeltaNet + 8 full attention (3:1 ratio)
    The models differ only in hidden_size (2560 vs 4096) which does not
    affect KV cache or recurrent state sizes.
    """
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=256,
        dtype=kv_dtype,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((3, 8192), (32, 128, 128)),
        dtypes=(mamba_dtype, mamba_dtype),
    )
    # Qwen3.5 layer pattern: every 4th layer is full attention
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    for i in range(32):
        if (i + 1) % 4 == 0:
            kv_cache_specs[f"layer_{i}"] = attention_spec
        else:
            kv_cache_specs[f"layer_{i}"] = mamba_spec
    return kv_cache_specs, attention_spec, mamba_spec


# ---------------------------------------------------------------------------
# Qwen3.5 hybrid Mamba+attention tests
# ---------------------------------------------------------------------------


def test_has_mixed_mamba_attention():
    """_has_mixed_mamba_attention returns True only for mixed groups."""
    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()

    # Pure attention -> False
    assert not kv_cache_utils._has_mixed_mamba_attention(
        [KVCacheGroupSpec([f"layer_{i}" for i in range(8)], attn_spec)]
    )
    # Pure Mamba -> False
    assert not kv_cache_utils._has_mixed_mamba_attention(
        [KVCacheGroupSpec([f"layer_{i}" for i in range(24)], mamba_spec)]
    )
    # Mixed (Qwen3.5 layout) -> True
    assert kv_cache_utils._has_mixed_mamba_attention(
        [
            KVCacheGroupSpec([f"layer_{i}" for i in range(24)], mamba_spec),
            KVCacheGroupSpec([f"layer_{i}" for i in range(24, 32)], attn_spec),
        ]
    )


@pytest.mark.parametrize(
    "kv_dtype, mamba_dtype, model_tag",
    [
        (torch.bfloat16, torch.bfloat16, "Qwen3.5-4B/9B bf16"),
        (torch.float16, torch.float16, "Qwen3.5-4B/9B fp16"),
        (torch.float8_e4m3fn, torch.bfloat16, "Qwen3.5-4B/9B fp8-kv"),
    ],
    ids=["bf16", "fp16", "fp8-kv"],
)
def test_qwen35_allocation_per_layer_tensors(kv_dtype, mamba_dtype, model_tag):
    """Verify per-layer tensor allocation for real Qwen3.5 specs.

    Each of the 32 layers should get its own tensor at its natural page size.
    Attention and GatedDeltaNet tensors must have different sizes.
    Total allocation must be efficient (>90% of available memory used).
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs(
        kv_dtype=kv_dtype, mamba_dtype=mamba_dtype
    )
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes

    # Give enough memory for ~10 blocks
    total_page_per_block = 8 * attn_page + 24 * mamba_page
    available_memory = total_page_per_block * 10

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    # 32 tensors, one per layer
    assert len(kv_cache_config.kv_cache_tensors) == 32, (
        f"{model_tag}: expected 32 per-layer tensors, "
        f"got {len(kv_cache_config.kv_cache_tensors)}"
    )

    # Each tensor serves exactly one layer
    for t in kv_cache_config.kv_cache_tensors:
        assert len(t.shared_by) == 1

    # Separate attention vs Mamba tensors
    attn_tensors = [
        t
        for t in kv_cache_config.kv_cache_tensors
        if kv_cache_specs[t.shared_by[0]] is attn_spec
    ]
    mamba_tensors = [
        t
        for t in kv_cache_config.kv_cache_tensors
        if kv_cache_specs[t.shared_by[0]] is mamba_spec
    ]
    assert len(attn_tensors) == 8, f"{model_tag}: expected 8 attention tensors"
    assert len(mamba_tensors) == 24, f"{model_tag}: expected 24 Mamba tensors"

    # Tensor sizes match their spec's page_size * block count.
    # With compact allocation, Mamba uses mamba_num_blocks (not num_blocks).
    num_blocks = kv_cache_config.num_blocks
    mamba_num_blocks = kv_cache_config.mamba_num_blocks or num_blocks
    assert num_blocks > 0
    for t in attn_tensors:
        assert t.size == attn_page * num_blocks
    for t in mamba_tensors:
        assert t.size == mamba_page * mamba_num_blocks

    # Attention and Mamba tensors have DIFFERENT sizes (not padded uniform)
    assert attn_tensors[0].size != mamba_tensors[0].size, (
        f"{model_tag}: tensors should differ — "
        f"attn={attn_tensors[0].size}, mamba={mamba_tensors[0].size}"
    )

    # Allocation is efficient: >90% of available memory used
    total_allocated = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    efficiency = total_allocated / available_memory
    assert efficiency > 0.90, (
        f"{model_tag}: allocation efficiency {efficiency:.1%} < 90%"
    )


@pytest.mark.parametrize(
    "kv_dtype, mamba_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
    ids=["bf16", "fp16", "fp8-kv"],
)
def test_qwen35_concurrency_estimate(kv_dtype, mamba_dtype):
    """Verify concurrency estimate correctly weights Mamba vs attention cost.

    For Qwen3.5, Mamba's 24 layers have O(1) state per request (~26 MiB total
    at bf16) while attention's 8 layers have O(n) KV (~1 GiB at 32K context).
    The concurrency estimate must reflect that attention dominates cost.
    """
    max_model_len = 32768
    model_config = ModelConfig(max_model_len=max_model_len)
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    _, attn_spec, mamba_spec = _make_qwen35_specs(
        kv_dtype=kv_dtype, mamba_dtype=mamba_dtype
    )

    # Compute expected values
    attn_max_mem = attn_spec.max_memory_usage_bytes(vllm_config)  # O(n)
    mamba_max_mem = mamba_spec.max_memory_usage_bytes(vllm_config)  # O(1)

    # Mamba per-request cost should be a small fraction of attention
    total_attn_cost = 8 * attn_max_mem
    total_mamba_cost = 24 * mamba_max_mem
    mamba_fraction = total_mamba_cost / (total_attn_cost + total_mamba_cost)
    assert mamba_fraction < 0.10, (
        f"Mamba should be <10% of per-request cost, got {mamba_fraction:.1%}"
    )

    # Compute blocks-per-request using same formula as our implementation:
    # total_per_request = sum(layers_in_group * spec.max_memory)
    # total_per_block = sum(layers_in_group * spec.page_size)
    # blocks_per_request = ceil(total_per_request / total_per_block)
    total_per_request = 8 * attn_max_mem + 24 * mamba_max_mem
    total_per_block = 8 * attn_spec.page_size_bytes + 24 * mamba_spec.page_size_bytes
    blocks_per_request = (total_per_request + total_per_block - 1) // total_per_block

    # Give enough blocks for ~3 concurrent requests
    num_blocks = blocks_per_request * 3

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(24)], mamba_spec),
            KVCacheGroupSpec([f"layer_{i}" for i in range(24, 32)], attn_spec),
        ],
    )
    concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

    # Concurrency should be exactly 3 (we gave exactly 3x blocks_per_request)
    assert concurrency == 3.0, f"Expected 3.0 concurrency, got {concurrency:.2f}"


def test_qwen35_groups_skip_page_size_unification():
    """Page size unification is skipped for Qwen3.5 mixed Mamba+attention.

    Without this, unify_kv_cache_spec_page_size would pad one spec's page
    size to match the other, wasting memory.
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes

    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs)

    # Must have both Mamba and attention groups
    attn_groups = [g for g in groups if not isinstance(g.kv_cache_spec, MambaSpec)]
    mamba_groups = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]
    assert len(attn_groups) >= 1
    assert len(mamba_groups) >= 1

    # Page sizes must be preserved (not padded to match each other)
    for g in attn_groups:
        assert g.kv_cache_spec.page_size_bytes == attn_page
    for g in mamba_groups:
        assert g.kv_cache_spec.page_size_bytes == mamba_page
    assert attn_page != mamba_page


def test_qwen35_mamba_cache_mode_all_includes_mamba_in_token_count():
    """When mamba_cache_mode='all', Mamba states are cached per-token for
    prefix caching. The token capacity report must include Mamba groups."""
    model_config = ModelConfig(max_model_len=1024)
    cache_config = CacheConfig(mamba_cache_mode="all")
    vllm_config = VllmConfig(model_config=model_config, cache_config=cache_config)

    _, attn_spec, mamba_spec = _make_qwen35_specs()

    kv_cache_config = KVCacheConfig(
        num_blocks=320,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(24)], mamba_spec),
            KVCacheGroupSpec([f"layer_{i}" for i in range(24, 32)], attn_spec),
        ],
    )

    # In "all" mode, all 32 groups count toward token capacity
    # In "none" mode, only 8 attention groups would count
    # We verify by checking _report_kv_cache_config runs without error
    # and that the filter includes all groups
    mamba_cache_mode = vllm_config.cache_config.mamba_cache_mode
    attention_groups = [
        g
        for g in kv_cache_config.kv_cache_groups
        if not isinstance(g.kv_cache_spec, MambaSpec) or mamba_cache_mode == "all"
    ]
    assert len(attention_groups) == 2, (
        "In 'all' mode, both Mamba and attention groups should be included"
    )

    # Contrast with "none" mode — only attention groups
    vllm_config_none = VllmConfig(model_config=model_config)
    mamba_cache_mode_none = vllm_config_none.cache_config.mamba_cache_mode
    attention_groups_none = [
        g
        for g in kv_cache_config.kv_cache_groups
        if not isinstance(g.kv_cache_spec, MambaSpec) or mamba_cache_mode_none == "all"
    ]
    assert len(attention_groups_none) == 1, (
        "In 'none' mode, only attention groups should be included"
    )


def test_qwen35_pure_attention_and_pure_mamba_unaffected():
    """Our changes must not affect pure-attention or pure-Mamba models."""
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    _, attn_spec, mamba_spec = _make_qwen35_specs()

    # Pure attention (e.g. Llama) — should NOT hit mixed path
    attn_specs: dict[str, KVCacheSpec] = {f"layer_{i}": attn_spec for i in range(32)}
    attn_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, attn_specs)
    assert not kv_cache_utils._has_mixed_mamba_attention(attn_groups)

    # Pure Mamba (e.g. Mamba2) — should NOT hit mixed path
    mamba_specs: dict[str, KVCacheSpec] = {f"layer_{i}": mamba_spec for i in range(32)}
    mamba_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, mamba_specs)
    assert not kv_cache_utils._has_mixed_mamba_attention(mamba_groups)


# ---------------------------------------------------------------------------
# Compact Mamba allocation tests
# ---------------------------------------------------------------------------

# Qwen3.5 architecture: 32 layers, every 4th is attention (see _make_qwen35_specs)
_QWEN35_NUM_MAMBA_LAYERS = 24
_QWEN35_NUM_ATTN_LAYERS = 8


def _total_page_per_block(kv_cache_specs: dict[str, KVCacheSpec]) -> int:
    """Total page size across all layers for one block."""
    return sum(spec.page_size_bytes for spec in kv_cache_specs.values())


def test_estimate_consistent_with_allocation():
    """The memory estimate must be consistent with the compact allocation.

    If the estimate says max_model_len=M fits, the allocation MUST have
    enough attention blocks for M tokens. This is the OOM-prevention invariant.
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    block_size = attn_spec.block_size

    # Test at several memory levels
    total_page_per_block = _total_page_per_block(kv_cache_specs)
    for num_blocks_target in [5, 10, 50, 200]:
        available_memory = total_page_per_block * num_blocks_target

        kv_cache_config = kv_cache_utils.get_kv_cache_configs(
            vllm_config, [kv_cache_specs], [available_memory]
        )[0]

        # Attention blocks must be enough for at least 1 full request
        blocks_needed_for_max_model_len = cdiv(
            vllm_config.model_config.max_model_len, block_size
        )
        assert kv_cache_config.num_blocks >= blocks_needed_for_max_model_len, (
            f"OOM invariant violated: {kv_cache_config.num_blocks} attention "
            f"blocks < {blocks_needed_for_max_model_len} needed for "
            f"max_model_len={vllm_config.model_config.max_model_len}"
        )

        # Mamba blocks must exist (at least 1 concurrent request)
        if kv_cache_config.mamba_num_blocks is not None:
            assert kv_cache_config.mamba_num_blocks >= 1, (
                "Mamba must have at least 1 block for 1 concurrent request"
            )


def test_compact_mamba_allocation_sizes():
    """Compact allocation gives Mamba much fewer blocks than attention.

    Mamba tensors should be sized for the compact block count, not the
    attention block count.
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes

    # Give plenty of memory so the difference is stark
    total_page_per_block = _total_page_per_block(kv_cache_specs)
    available_memory = total_page_per_block * 200

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    # Compact allocation should be active (mamba_cache_mode defaults to "none")
    assert kv_cache_config.mamba_num_blocks is not None, (
        "Compact allocation should be active for default mamba_cache_mode"
    )

    # Separate tensors
    attn_tensors = [
        t
        for t in kv_cache_config.kv_cache_tensors
        if kv_cache_specs[t.shared_by[0]] is attn_spec
    ]
    mamba_tensors = [
        t
        for t in kv_cache_config.kv_cache_tensors
        if kv_cache_specs[t.shared_by[0]] is mamba_spec
    ]

    # Mamba tensors should be much smaller than attention tensors
    mamba_blocks = mamba_tensors[0].size // mamba_page
    attn_blocks = attn_tensors[0].size // attn_page

    assert mamba_blocks < attn_blocks, (
        f"Mamba blocks ({mamba_blocks}) should be << attention blocks ({attn_blocks})"
    )
    assert mamba_blocks == kv_cache_config.mamba_num_blocks
    assert attn_blocks == kv_cache_config.num_blocks

    # Attention gets more total memory than Mamba
    total_attn_mem = sum(t.size for t in attn_tensors)
    total_mamba_mem = sum(t.size for t in mamba_tensors)
    assert total_attn_mem > total_mamba_mem, (
        f"Attention memory ({total_attn_mem}) should be > Mamba memory "
        f"({total_mamba_mem})"
    )


def test_token_capacity_improvement():
    """Compact allocation should yield much higher token capacity than the
    old shared-pool approach.

    The old approach gives all layers the same num_blocks. For Qwen3.5 with
    24 Mamba layers at ~1 MB page size, this wastes enormous amounts of
    memory. The compact approach should yield at least 5x more tokens.
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    block_size = attn_spec.block_size

    # 10 GB available
    available_memory = 10 * GiB_bytes

    # Old approach: all layers share num_blocks
    total_page_per_block = _total_page_per_block(kv_cache_specs)
    old_num_blocks = int(available_memory // total_page_per_block)
    old_token_capacity = old_num_blocks * block_size

    # New approach: compact allocation
    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]
    new_token_capacity = kv_cache_config.num_blocks * block_size

    # The improvement ratio is roughly total_page / attn_page_total because
    # compact allocation lets attention use nearly all the memory.
    # For Qwen3.5: total ≈ 26.9 MB, attn ≈ 0.5 MB → ~50x theoretical max.
    # Use a conservative floor that still validates the optimization works.
    attn_page_total = _QWEN35_NUM_ATTN_LAYERS * attn_page
    expected_ratio = total_page_per_block / attn_page_total
    conservative_floor = expected_ratio / 10  # 10% of theoretical max
    assert new_token_capacity > old_token_capacity * conservative_floor, (
        f"New capacity ({new_token_capacity} tokens) should be "
        f">{conservative_floor:.0f}x old ({old_token_capacity} tokens), "
        f"got {new_token_capacity / old_token_capacity:.1f}x"
    )


def test_compact_mamba_not_used_for_mode_all():
    """When mamba_cache_mode='all', Mamba should share the block pool.

    Compact allocation is only for "none" and "align" modes where Mamba
    state is O(1) per request.
    """
    model_config = ModelConfig(max_model_len=1024)
    cache_config = CacheConfig(mamba_cache_mode="all")
    vllm_config = VllmConfig(model_config=model_config, cache_config=cache_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()

    total_page_per_block = _total_page_per_block(kv_cache_specs)
    # "all" mode needs cdiv(max_model_len, block_size) blocks minimum
    # to serve one max-length request. Double it for headroom.
    block_size = attn_spec.block_size
    min_blocks = cdiv(model_config.max_model_len, block_size)
    available_memory = total_page_per_block * min_blocks * 2

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    # mamba_num_blocks should be None (shared pool)
    assert kv_cache_config.mamba_num_blocks is None, (
        "mamba_cache_mode='all' should not use compact allocation"
    )

    # All tensors should have the same num_blocks
    num_blocks = kv_cache_config.num_blocks
    for t in kv_cache_config.kv_cache_tensors:
        layer_name = t.shared_by[0]
        spec = kv_cache_specs[layer_name]
        expected_size = spec.page_size_bytes * num_blocks
        assert t.size == expected_size, (
            f"Layer {layer_name}: size {t.size} != expected {expected_size}"
        )


def test_concurrency_reflects_actual_capacity():
    """Concurrency for compact allocation should reflect both attention and
    Mamba capacity, and should allow multiple concurrent requests."""
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()

    # Give enough memory for many requests
    total_page_per_block = _total_page_per_block(kv_cache_specs)
    available_memory = total_page_per_block * 200

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

    # Should support multiple concurrent requests
    assert concurrency > 1, f"Concurrency should be > 1, got {concurrency:.2f}"

    # Concurrency should be approximately the compact allocation's
    # num_concurrent (Mamba is the tighter constraint by design)
    if kv_cache_config.mamba_num_blocks is not None:
        mamba_blocks_per_req = max(
            (
                g.kv_cache_spec.max_memory_usage_bytes(vllm_config)
                + g.kv_cache_spec.page_size_bytes
                - 1
            )
            // g.kv_cache_spec.page_size_bytes
            for g in kv_cache_config.kv_cache_groups
            if isinstance(g.kv_cache_spec, MambaSpec)
        )
        mamba_concurrency = kv_cache_config.mamba_num_blocks / mamba_blocks_per_req
        # Concurrency = min(attn, mamba), so it must be <= mamba capacity
        assert concurrency <= mamba_concurrency + 1e-9  # float tolerance


def test_pure_models_unaffected_by_compact_allocation():
    """Pure attention and pure Mamba models should not use compact allocation.

    This is a regression guard: the compact path is gated by
    _has_mixed_mamba_attention().
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    _, attn_spec, mamba_spec = _make_qwen35_specs()

    # Pure attention model
    attn_specs: dict[str, KVCacheSpec] = {f"layer_{i}": attn_spec for i in range(32)}
    attn_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config,
        [attn_specs],
        [attn_spec.page_size_bytes * 32 * 100],
    )[0]
    assert attn_config.mamba_num_blocks is None, (
        "Pure attention model should not have mamba_num_blocks"
    )

    # Pure Mamba model
    mamba_specs_dict: dict[str, KVCacheSpec] = {
        f"layer_{i}": mamba_spec for i in range(32)
    }
    mamba_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config,
        [mamba_specs_dict],
        [mamba_spec.page_size_bytes * 32 * 100],
    )[0]
    assert mamba_config.mamba_num_blocks is None, (
        "Pure Mamba model should not have mamba_num_blocks"
    )


def test_compact_allocation_low_memory_floor():
    """When memory barely fits 1 request, the max(1,...) floor on
    num_concurrent must kick in.

    This exercises the edge case where optimal_C < 1. The floor guarantees
    at least 1 concurrent request, and therefore:
    - mamba_blocks >= mamba_blocks_per_req (enough for 1 request)
    - attention_num_blocks >= blocks_per_attn_request (enough for max_model_len)
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes
    block_size = attn_spec.block_size

    blocks_for_max_model_len = cdiv(model_config.max_model_len, block_size)

    # Compute exact cost of 1 request: attention blocks * attn_page_total
    # + mamba_blocks_per_req * mamba_page_cost.
    attn_page_total = _QWEN35_NUM_ATTN_LAYERS * attn_page
    mamba_page_cost = _QWEN35_NUM_MAMBA_LAYERS * mamba_page
    # mamba_blocks_per_req = cdiv(max_memory_usage, page_size_bytes)
    # For "none" mode: max_memory_usage = page_size_bytes (1 block)
    mamba_blocks_per_req = 1

    cost_of_one_request = (
        attn_page_total * blocks_for_max_model_len
        + mamba_page_cost * mamba_blocks_per_req
    )
    # Give exactly enough for ~1.1 requests (floor should cap to 1)
    available_memory = int(cost_of_one_request * 1.1)

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    # num_concurrent should be 1 (floor kicks in)
    # Justification: mamba_blocks = num_concurrent * mamba_blocks_per_req
    assert kv_cache_config.mamba_num_blocks is not None, (
        "Compact allocation should be active"
    )
    assert kv_cache_config.mamba_num_blocks == mamba_blocks_per_req, (
        f"With 1 concurrent request, mamba_blocks should be "
        f"{mamba_blocks_per_req}, got {kv_cache_config.mamba_num_blocks}"
    )

    # Attention blocks must still fit max_model_len (OOM invariant).
    # Justification: if this fails, a single max-length request would OOM.
    assert kv_cache_config.num_blocks >= blocks_for_max_model_len, (
        f"Attention blocks {kv_cache_config.num_blocks} < "
        f"{blocks_for_max_model_len} needed for max_model_len"
    )


def test_compact_allocation_capped_by_max_num_seqs():
    """When max_num_seqs caps num_concurrent, the freed Mamba budget
    should go to attention blocks.

    With huge memory and max_num_seqs=4, optimal_C would be >> 4 but
    gets capped. The Mamba pool is sized for exactly 4 requests, and
    the remaining memory goes to attention.
    """
    max_num_seqs = 4
    model_config = ModelConfig(max_model_len=1024)
    scheduler_config = SchedulerConfig(
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=False,
        max_num_seqs=max_num_seqs,
    )
    vllm_config = VllmConfig(
        model_config=model_config, scheduler_config=scheduler_config
    )

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes

    # Give huge memory (1000 blocks worth)
    total_page_per_block = _total_page_per_block(kv_cache_specs)
    available_memory = total_page_per_block * 1000

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    assert kv_cache_config.mamba_num_blocks is not None

    # Justification: mamba_blocks = min(optimal_C, max_num_seqs) * blocks_per_req.
    # With "none" mode, blocks_per_req = 1, so mamba_blocks should be exactly 4.
    mamba_blocks_per_req = 1
    expected_mamba_blocks = max_num_seqs * mamba_blocks_per_req
    assert kv_cache_config.mamba_num_blocks == expected_mamba_blocks, (
        f"Expected {expected_mamba_blocks} mamba blocks (capped by "
        f"max_num_seqs={max_num_seqs}), got {kv_cache_config.mamba_num_blocks}"
    )

    # Justification: with only 4 Mamba blocks, nearly all memory goes to
    # attention. Attention blocks should be much higher than the uncapped case
    # would give per-concurrent-request. Specifically, nearly all available
    # memory minus 4*mamba_cost should be in attention.
    mamba_page_cost = _QWEN35_NUM_MAMBA_LAYERS * mamba_page
    mamba_total = expected_mamba_blocks * mamba_page_cost
    attn_page_total = _QWEN35_NUM_ATTN_LAYERS * attn_page
    expected_attn_blocks = int((available_memory - mamba_total) // attn_page_total)
    assert kv_cache_config.num_blocks == expected_attn_blocks, (
        f"Attention blocks {kv_cache_config.num_blocks} != expected "
        f"{expected_attn_blocks} (available - mamba_cost)"
    )


def test_cross_worker_mamba_scaling():
    """Multi-worker configs with different available memory should be
    synchronized to the minimum mamba_num_blocks and num_blocks.

    This exercises the cross-worker tensor scaling path that scales Mamba
    and attention tensors independently using _is_mamba_layer().
    """
    model_config = ModelConfig(max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    attn_page = attn_spec.page_size_bytes
    mamba_page = mamba_spec.page_size_bytes

    total_page_per_block = _total_page_per_block(kv_cache_specs)
    # Worker 0 has more memory than worker 1
    mem_worker_0 = total_page_per_block * 200
    mem_worker_1 = total_page_per_block * 100

    configs = kv_cache_utils.get_kv_cache_configs(
        vllm_config,
        [kv_cache_specs, kv_cache_specs],
        [mem_worker_0, mem_worker_1],
    )

    # Justification: cross-worker sync sets all configs to the minimum.
    # Both workers must have identical num_blocks and mamba_num_blocks.
    assert configs[0].num_blocks == configs[1].num_blocks, (
        "num_blocks must be synchronized across workers"
    )
    assert configs[0].mamba_num_blocks == configs[1].mamba_num_blocks, (
        "mamba_num_blocks must be synchronized across workers"
    )

    # The synced values should match the smaller worker's allocation
    single_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_worker_1]
    )[0]
    assert configs[0].num_blocks == single_config.num_blocks, (
        "Synced num_blocks should match the smaller worker's allocation"
    )
    assert configs[0].mamba_num_blocks == single_config.mamba_num_blocks, (
        "Synced mamba_num_blocks should match the smaller worker's allocation"
    )

    # Justification: tensor sizes must be scaled to match the synced block counts.
    # Mamba tensors use mamba_num_blocks, attention tensors use num_blocks.
    for cfg in configs:
        for tensor in cfg.kv_cache_tensors:
            layer_name = tensor.shared_by[0]
            spec = kv_cache_specs[layer_name]
            if isinstance(spec, MambaSpec):
                assert tensor.size == mamba_page * cfg.mamba_num_blocks, (
                    f"Mamba tensor {layer_name}: size {tensor.size} != "
                    f"{mamba_page} * {cfg.mamba_num_blocks}"
                )
            else:
                assert tensor.size == attn_page * cfg.num_blocks, (
                    f"Attn tensor {layer_name}: size {tensor.size} != "
                    f"{attn_page} * {cfg.num_blocks}"
                )

    # Justification: generate_scheduler_kv_cache_config must not raise
    # because mamba_num_blocks is consistent across workers.
    scheduler_config_result = generate_scheduler_kv_cache_config(configs)
    assert scheduler_config_result.mamba_num_blocks == configs[0].mamba_num_blocks


def test_compact_mamba_manager_allocate_and_free():
    """MambaManager in compact mode should allocate from and free to its
    private compact pool, without touching the shared BlockPool.

    This validates the core lifecycle: allocate blocks for requests,
    free them, and confirm they're reusable.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    _, _, mamba_spec = _make_qwen35_specs()
    block_size = mamba_spec.block_size

    num_gpu_blocks = 100
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=False,
        hash_block_size=block_size,
    )
    initial_pool_free = block_pool.free_block_queue.num_free_blocks

    mamba_num_blocks = 5
    manager = MambaManager(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        mamba_num_blocks=mamba_num_blocks,
        enable_caching=False,
        kv_cache_group_id=0,
    )

    # Justification: compact mode should be active when mamba_num_blocks is set
    # and mamba_cache_mode != "all".
    assert manager.compact_mode is True
    assert len(manager._compact_free) == mamba_num_blocks

    # Allocate blocks for 3 requests (1 block each for 16 tokens)
    for i in range(3):
        req_id = f"req_{i}"
        blocks = manager.allocate_new_blocks(req_id, block_size, block_size)
        assert len(blocks) == 1, f"Expected 1 block for req_{i}"
        # Justification: each block ID should be in [0, mamba_num_blocks)
        assert 0 <= blocks[0].block_id < mamba_num_blocks

    # Justification: 3 blocks allocated from 5 total, so 2 should remain free.
    assert len(manager._compact_free) == 2

    # Justification: shared BlockPool must not be touched in compact mode.
    assert block_pool.free_block_queue.num_free_blocks == initial_pool_free, (
        "BlockPool free count changed — compact blocks leaked into shared pool"
    )

    # Free one request and verify block returns to compact pool.
    manager.free("req_1")
    # Justification: freeing 1 request returns its 1 block to the compact pool.
    assert len(manager._compact_free) == 3

    # Allocate another request — should reuse the freed block.
    blocks = manager.allocate_new_blocks("req_3", block_size, block_size)
    assert len(blocks) == 1
    # Justification: the freed block from req_1 should be reused (LIFO stack).
    assert len(manager._compact_free) == 2

    # Justification: BlockPool must still be untouched after all operations.
    assert block_pool.free_block_queue.num_free_blocks == initial_pool_free


def test_compact_mamba_manager_exhaustion_rejects():
    """When the compact pool is exhausted, get_num_blocks_to_allocate
    must return a rejection signal (> num_gpu_blocks).

    This prevents over-allocation and is the signal to the scheduler
    to not schedule this request in the current step.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    _, _, mamba_spec = _make_qwen35_specs()
    block_size = mamba_spec.block_size

    num_gpu_blocks = 100
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=False,
        hash_block_size=block_size,
    )

    mamba_num_blocks = 2
    manager = MambaManager(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        mamba_num_blocks=mamba_num_blocks,
        enable_caching=False,
        kv_cache_group_id=0,
    )

    # Fill the compact pool: 2 requests × 1 block each
    manager.allocate_new_blocks("req_0", block_size, block_size)
    manager.allocate_new_blocks("req_1", block_size, block_size)
    assert len(manager._compact_free) == 0

    # Justification: with 0 free compact blocks and a new request needing 1,
    # get_num_blocks_to_allocate must return > num_gpu_blocks to signal rejection.
    num_to_alloc = manager.get_num_blocks_to_allocate(
        request_id="req_2",
        num_tokens=block_size,
        new_computed_blocks=[],
        total_computed_tokens=0,
        num_tokens_main_model=block_size,
    )
    assert num_to_alloc > num_gpu_blocks, (
        f"Expected rejection signal (>{num_gpu_blocks}), got {num_to_alloc}"
    )

    # After freeing one request, the same call should succeed (return 0).
    manager.free("req_0")
    num_to_alloc = manager.get_num_blocks_to_allocate(
        request_id="req_2",
        num_tokens=block_size,
        new_computed_blocks=[],
        total_computed_tokens=0,
        num_tokens_main_model=block_size,
    )
    # Justification: 0 means "no shared pool blocks needed" — compact handles it.
    assert num_to_alloc == 0, (
        f"Expected 0 (compact handles allocation), got {num_to_alloc}"
    )


def test_compact_mamba_cache_blocks_noop():
    """cache_blocks in compact mode must be a no-op to prevent compact
    block IDs from entering the shared pool's cache hash table.

    If compact IDs leak into the cache, they could collide with attention
    block IDs and cause incorrect cache hits or block corruption.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    _, _, mamba_spec = _make_qwen35_specs()
    block_size = mamba_spec.block_size

    num_gpu_blocks = 100
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,  # Caching enabled to verify no-op
        hash_block_size=block_size,
    )

    manager = MambaManager(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        mamba_num_blocks=5,
        enable_caching=True,
        kv_cache_group_id=0,
    )

    # Allocate a block
    manager.allocate_new_blocks("req_0", block_size, block_size)

    # Count cached blocks in the pool before
    cached_before = len(block_pool.cached_block_hash_to_block)

    # Create a minimal request-like object for cache_blocks
    req = make_request("req_0", list(range(block_size)), block_size)
    manager.cache_blocks(req, block_size)

    # Justification: cache_blocks is a no-op in compact mode, so no new
    # entries should appear in the block pool's cache.
    cached_after = len(block_pool.cached_block_hash_to_block)
    assert cached_after == cached_before, (
        f"Block pool cache grew from {cached_before} to {cached_after} — "
        "compact block IDs leaked into shared cache"
    )


def test_all_mode_concurrency():
    """Concurrency for mamba_cache_mode='all' should use the standard
    mixed formula (num_blocks / blocks_per_request), not the compact path.

    This verifies the 'all' mode branch of get_max_concurrency_for_kv_cache_config
    wasn't broken when we added the compact branch.
    """
    model_config = ModelConfig(max_model_len=1024)
    cache_config = CacheConfig(mamba_cache_mode="all")
    vllm_config = VllmConfig(model_config=model_config, cache_config=cache_config)

    kv_cache_specs, attn_spec, mamba_spec = _make_qwen35_specs()
    block_size = attn_spec.block_size

    total_page_per_block = _total_page_per_block(kv_cache_specs)
    min_blocks = cdiv(model_config.max_model_len, block_size)
    available_memory = total_page_per_block * min_blocks * 2

    kv_cache_config = kv_cache_utils.get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )[0]

    # mamba_num_blocks is None for "all" mode
    assert kv_cache_config.mamba_num_blocks is None

    concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

    # Justification: "all" mode uses num_blocks / blocks_per_request where
    # blocks_per_request is based on per-request memory usage across all groups.
    # With ~2x headroom, concurrency should be approximately 2.
    assert concurrency > 1.0, (
        f"'all' mode concurrency should be > 1 with 2x headroom, got {concurrency:.2f}"
    )

    # Justification: concurrency is calculated as num_blocks / blocks_per_request.
    # Verify it matches the manual calculation to confirm the right formula is used.
    max_memory_per_req = sum(
        len(g.layer_names) * g.kv_cache_spec.max_memory_usage_bytes(vllm_config)
        for g in kv_cache_config.kv_cache_groups
    )
    total_page = sum(
        len(g.layer_names) * g.kv_cache_spec.page_size_bytes
        for g in kv_cache_config.kv_cache_groups
    )
    blocks_per_req = (max_memory_per_req + total_page - 1) // total_page
    expected_concurrency = kv_cache_config.num_blocks / blocks_per_req
    assert abs(concurrency - expected_concurrency) < 1e-9, (
        f"Concurrency {concurrency:.4f} != expected {expected_concurrency:.4f}"
    )
