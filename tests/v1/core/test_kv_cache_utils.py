# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import GiB_bytes, sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
# disable yapf here as it formats differently than isort such that both fail
# yapf: disable
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue, KVCacheBlock, PrefixCachingMetrics,
    estimate_max_model_len, generate_block_hash_extra_keys,
    get_kv_cache_config, get_max_concurrency_for_kv_cache_config,
    hash_block_tokens, hash_request_tokens, unify_kv_cache_configs)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        SlidingWindowSpec)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

# yapf: enable


def make_request(request_id,
                 prompt_token_ids,
                 mm_positions=None,
                 mm_hashes=None,
                 cache_salt=None):
    if mm_positions is None:
        multi_modal_inputs = None
    else:
        multi_modal_inputs = [MultiModalKwargs({})] * len(mm_positions)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        multi_modal_inputs=multi_modal_inputs,
        multi_modal_hashes=mm_hashes,
        multi_modal_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
        cache_salt=cache_salt,
    )


def new_kv_cache_spec(block_size=16,
                      num_kv_heads=2,
                      head_size=64,
                      dtype=torch.float32,
                      use_mla=False,
                      sliding_window=None):
    return FullAttentionSpec(block_size=block_size,
                             num_kv_heads=num_kv_heads,
                             head_size=head_size,
                             dtype=dtype,
                             use_mla=use_mla,
                             sliding_window=sliding_window)


def new_sliding_window_spec(block_size=16,
                            num_kv_heads=2,
                            head_size=64,
                            dtype=torch.float32,
                            use_mla=False,
                            sliding_window=1):
    return SlidingWindowSpec(block_size=block_size,
                             num_kv_heads=num_kv_heads,
                             head_size=head_size,
                             dtype=dtype,
                             use_mla=use_mla,
                             sliding_window=sliding_window)


def test_none_hash(monkeypatch):
    import vllm.v1.core.kv_cache_utils

    # case 1: PYTHONHASHSEED is not set, use random
    with monkeypatch.context() as m:
        m.delenv('PYTHONHASHSEED', raising=False)
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, int)
        assert reloaded_kv_cache_utils.NONE_HASH != 0

    # case 2: PYTHONHASHSEED is set, use the seed
    with monkeypatch.context() as m:
        m.setenv('PYTHONHASHSEED', 'python hash seed')
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, int)
        assert sha256('python hash seed') == reloaded_kv_cache_utils.NONE_HASH


def test_kv_cache_block():
    import vllm.v1.core.kv_cache_utils

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
    block_hash = vllm.v1.core.kv_cache_utils.BlockHash(hash_value=123,
                                                       token_ids=(1, 2, 3))
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
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=10, length=5),
        ],
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
    assert extra_keys is None
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


def test_generate_block_hash_extra_keys_cache_salt():
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
        cache_salt="salt",
    )

    # salt is added for the first token
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 1, 0)
    assert extra_keys == ('salt', )
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 10, 0)
    assert extra_keys == ('salt', )

    # no salt added for other tokens
    extra_keys, _ = generate_block_hash_extra_keys(request, 1, 2, 0)
    assert extra_keys is None
    extra_keys, _ = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None

    # works together with other extra keys
    request_mm = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
        ],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(
        request_mm, 0, 5, 0)
    assert extra_keys == ("hash1", "salt")
    assert next_mm_idx == 1


@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_hash_block_tokens(hash_fn):
    import vllm.v1.core.kv_cache_utils
    parent_block_hash = 123
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(hash_fn, parent_block_hash,
                                   curr_block_token_ids, extra_keys)
    assert isinstance(block_hash, vllm.v1.core.kv_cache_utils.BlockHash)
    assert block_hash.hash_value == hash_fn(
        (parent_block_hash, curr_block_token_ids, extra_keys))
    assert block_hash.token_ids == curr_block_token_ids
    assert block_hash.extra_keys == extra_keys


@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_hash_request_tokens(hash_fn):
    import vllm.v1.core.kv_cache_utils
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    block_size = 3
    block_hashes = hash_request_tokens(hash_fn, block_size, request)

    assert len(block_hashes) == 2
    assert isinstance(block_hashes[0], vllm.v1.core.kv_cache_utils.BlockHash)
    assert isinstance(block_hashes[1], vllm.v1.core.kv_cache_utils.BlockHash)

    # Check the first block
    assert block_hashes[0].token_ids == (0, 1, 2)
    assert block_hashes[0].extra_keys == ("hash1", )

    # Check the second block
    assert block_hashes[1].token_ids == (3, 4, 5)
    assert block_hashes[1].extra_keys == ("hash2", )


@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_hash_tokens_different_mm_input(hash_fn):
    request1 = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )
    request2 = make_request(
        request_id=1,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash3", "hash2"],
    )
    block_size = 3
    block_hashes1 = hash_request_tokens(hash_fn, block_size, request1)
    block_hashes2 = hash_request_tokens(hash_fn, block_size, request2)
    assert block_hashes1[0] != block_hashes2[0]
    assert block_hashes1[1] != block_hashes2[1]


@pytest.mark.parametrize("hash_fn", [sha256, hash])
def test_hash_request_tokens_no_mm_inputs(hash_fn):
    request = make_request(
        request_id=0,
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    block_size = 3
    block_hashes = hash_request_tokens(hash_fn, block_size, request)

    assert len(block_hashes) == 2
    assert block_hashes[0].token_ids == (0, 1, 2)
    assert block_hashes[0].extra_keys is None
    assert block_hashes[1].token_ids == (3, 4, 5)
    assert block_hashes[1].extra_keys is None


def test_metrics():
    """
    Test the prefix caching metrics.
    """

    def stats(requests, queries, hits):
        return PrefixCacheStats(requests=requests, queries=queries, hits=hits)

    metrics = PrefixCachingMetrics(max_recent_requests=5)
    assert metrics.hit_rate == 0.0

    metrics.observe(stats(1, 20, 9))
    # 9 / 20 = 0.45
    assert metrics.hit_rate == 0.45

    metrics.observe(stats(4, 80, 16))

    # 25 / 100 = 0.25
    assert metrics.hit_rate == 0.25

    metrics.observe(stats(1, 10, 2))

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


def test_unify_kv_cache_configs():
    same_kv_cache_config = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=4)),
            ],
        ),
        KVCacheConfig(
            num_blocks=20,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=4)),
            ],
        ),
    ]
    unify_kv_cache_configs(same_kv_cache_config)
    assert same_kv_cache_config[0].num_blocks == 10
    assert same_kv_cache_config[1].num_blocks == 10

    need_sort_kv_cache_config = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=4)),
            ],
        ),
        KVCacheConfig(
            num_blocks=20,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=4)),
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
    ]

    unify_kv_cache_configs(need_sort_kv_cache_config)
    assert need_sort_kv_cache_config[0].num_blocks == 10
    assert need_sort_kv_cache_config[1].num_blocks == 10

    diff_kv_cache_config = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=4)),
            ],
        ),
        KVCacheConfig(
            num_blocks=20,
            kv_cache_tensors=[
                KVCacheTensor(size=100, shared_by=["layer1"]),
                KVCacheTensor(size=100, shared_by=["layer2"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
                KVCacheGroupSpec(["layer2"],
                                 new_kv_cache_spec(num_kv_heads=8)),
            ],
        ),
    ]
    with pytest.raises(AssertionError):
        unify_kv_cache_configs(diff_kv_cache_config)


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
            use_mla=full_spec.use_mla,
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
            different_sliding_window_layer_specs)

    same_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    ]
    merged_layer_spec = same_sliding_window_layer_specs[0].merge(
        same_sliding_window_layer_specs)
    assert merged_layer_spec.sliding_window == 1

    same_sliding_window_layer_spec_with_none = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=None),
    ]
    merged_layer_spec = same_sliding_window_layer_spec_with_none[0].merge(
        same_sliding_window_layer_spec_with_none)
    assert merged_layer_spec.sliding_window == 1


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"), [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ])
def test_estimate_max_model_len(model_id, max_model_len,
                                want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        task="generate",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(max_num_batched_tokens=32768)

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
            use_mla=False,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(vllm_config, kv_cache_spec,
                                               8 * GiB_bytes)
    assert estimated_max_len == want_estimated_max_len


def test_get_max_concurrency_for_kv_cache_config():
    # Create a VllmConfig
    model_id = "Qwen/Qwen1.5-7B"
    max_model_len = 16384
    model_config = ModelConfig(
        model_id,
        task="generate",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(max_num_batched_tokens=1024,
                                       enable_chunked_prefill=True)

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
    )

    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
        sliding_window=1024,
    )

    kv_cache_config_full_attention = KVCacheConfig(
        num_blocks=int(1024 * 1.5),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)],
                             full_attention_spec),
        ],
    )
    max_concurrency_full_attention = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_full_attention)
    assert max_concurrency_full_attention == 1.5

    kv_cache_config_sliding_window = KVCacheConfig(
        num_blocks=129 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)],
                             sliding_window_spec),
        ],
    )
    max_concurrency_sliding_window = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_sliding_window)
    assert max_concurrency_sliding_window == 3

    kv_cache_config_hybrid_model = KVCacheConfig(
        num_blocks=(1024 + 129) * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)],
                             full_attention_spec),
            KVCacheGroupSpec([f"layer_{i}" for i in range(32, 64)],
                             sliding_window_spec),
        ],
    )
    max_concurrency_hybrid_model = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_hybrid_model)
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
            KVCacheGroupSpec(["layer1"],
                             new_kv_cache_spec(block_size=block_size)),
        ],
    )

    request = make_request(
        request_id=0,
        prompt_token_ids=[],
        mm_positions=None,
        mm_hashes=None,
    )

    # Test case 1: Requires additional lookahead tokens
    kv_cache_manager = KVCacheManager(kv_cache_config=config,
                                      max_model_len=100)
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,  # Total required: 3+2=5 tokens
    )
    assert len(blocks.get_block_ids()[0]) == 2  # ceil(5/4)=2 blocks

    # Test case 2: With precomputed blocks
    kv_cache_manager = KVCacheManager(kv_cache_config=config,
                                      max_model_len=100)
    # required_blocks = ceil((3 + 2) /4) = 2
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,
    )
    assert len(blocks.get_block_ids()[0]) == 2

    # Test case 3: With precomputed blocks
    # required_blocks = ceil((3 + 4) / 4) = 2
    kv_cache_manager = KVCacheManager(kv_cache_config=config,
                                      max_model_len=100)
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=4,
    )
    assert len(blocks.get_block_ids()[0]) == 2


def test_get_kv_cache_config():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        'layer_1': new_kv_cache_spec(),
        'layer_2': new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_config(
        vllm_config, kv_cache_specs_full, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())
        ])

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        'layer_1': new_sliding_window_spec(),
        'layer_2': new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_config(
        vllm_config, kv_cache_specs_sliding, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ])

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        'layer_1': new_kv_cache_spec(),
        'layer_2': new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_config(
        vllm_config, kv_cache_specs_hybrid, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"],
                             new_kv_cache_spec(sliding_window=1)),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        'layer_1': new_kv_cache_spec(),
        'layer_2': new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_config(
        vllm_config, kv_cache_specs_hybrid, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 64,
                          shared_by=["layer_1", "layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        'layer_1': new_kv_cache_spec(),
        'layer_2': new_kv_cache_spec(),
        'layer_3': new_sliding_window_spec(),
        'layer_4': new_sliding_window_spec(),
        'layer_5': new_sliding_window_spec(),
        'layer_6': new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_config(
        vllm_config, kv_cache_specs_hybrid, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_1", "layer_3", "layer_5"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_2", "layer_4", "layer_6"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_4"],
                             new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_5", "layer_6"],
                             new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        'layer_1': new_kv_cache_spec(),
        'layer_2': new_kv_cache_spec(),
        'layer_3': new_kv_cache_spec(),
        'layer_4': new_sliding_window_spec(),
        'layer_5': new_sliding_window_spec(),
        'layer_6': new_sliding_window_spec(),
        'layer_7': new_sliding_window_spec(),
        'layer_8': new_sliding_window_spec(),
        'layer_9': new_sliding_window_spec(),
        'layer_10': new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_config(
        vllm_config, kv_cache_specs_hybrid, mem_per_block_per_layer * 3 * 32)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_4", "layer_7", "layer_10"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_2", "layer_5", "layer_8"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32,
                          shared_by=["layer_3", "layer_6", "layer_9"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"],
                             new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_4", "layer_5", "layer_6"],
                             new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_7", "layer_8", "layer_9"],
                             new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_10"], new_sliding_window_spec()),
        ],
    )

    # different hidden size, unimplemented
    kv_cache_specs_hybrid = {
        'layer_1': new_kv_cache_spec(head_size=128),
        'layer_2': new_kv_cache_spec(),
    }
    with pytest.raises(NotImplementedError):
        get_kv_cache_config(vllm_config, kv_cache_specs_hybrid,
                            mem_per_block_per_layer * 2 * 32)

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_config(
        vllm_config, kv_cache_specs_full, mem_per_block_per_layer * 2 * 32)
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 16,
                          shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 16,
                          shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())
        ])