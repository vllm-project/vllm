# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for optional per-request prefix-cache miss attribution (#60044).

These exercise the real KVCacheManager / BlockPool / Request path on CPU, so
they need no GPU, model weights or CUDA attention extensions.
"""

from math import lcm

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.prefix_cache_miss_attribution import RecentEvictions
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    group = KVCacheGroup(
        kv_cache_specs={"layer": spec},
        manager_config=FullAttentionManager.config(),
        block_size=spec.block_size,
        dcp_sharded=spec.dcp_sharded,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )


def make_manager(
    num_blocks: int,
    block_size: int = 16,
    enable_attribution: bool = True,
) -> KVCacheManager:
    config = make_config(block_size, num_blocks)
    return KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=lcm(*(g.block_size for g in config.kv_cache_groups)),
        hash_block_size=block_size,
        enable_caching=True,
        enable_prefix_cache_miss_attribution=enable_attribution,
    )


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int = 16,
    **kwargs,
) -> Request:
    mm_positions = kwargs.pop("mm_positions", None)
    mm_hashes = kwargs.pop("mm_hashes", None)
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_features.append(
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.dummy(),
                    mm_position=position,
                    identifier=identifier,
                    modality="image",
                )
            )

    sampling_params = kwargs.pop("sampling_params", None) or SamplingParams(
        max_tokens=17
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features or None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=kwargs.pop("lora_request", None),
        cache_salt=kwargs.pop("cache_salt", None),
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def warm_prefix(manager, request_id, tokens, **kwargs) -> None:
    warm = make_request(request_id, tokens, **kwargs)
    assert manager.allocate_slots(warm, len(tokens)) is not None
    manager.free(warm)


def attribution(manager, request):
    _, num_cached, _ = manager.get_computed_blocks(request)
    assert len(request.prefix_cache_miss_attributions) == 1
    return num_cached, request.prefix_cache_miss_attributions[0]


@pytest.mark.parametrize(
    "num_tokens, expected_hit", [(16, 0), (17, 16), (32, 16), (33, 32)]
)
def test_miss_attribution_lookup_limit_at_block_boundary(num_tokens, expected_hit):
    """The next block is recomputed for logits, not missed.

    The 16/17/32/33-token controls from the issue: with 32 tokens the second
    block is resident but never probed, so it must not be called a miss.
    """
    manager = make_manager(num_blocks=16)
    tokens = list(range(40))
    warm_prefix(manager, "warm", tokens)

    probe = make_request("probe", tokens[:num_tokens])
    num_cached, att = attribution(manager, probe)
    assert num_cached == expected_hit
    assert att.reason == "lookup_limit"
    assert att.block_index is None
    assert att.max_cache_hit_length == num_tokens - 1


def test_miss_attribution_no_match_for_unseen_prefix():
    manager = make_manager(num_blocks=16)
    warm_prefix(manager, "warm", list(range(40)))

    probe = make_request("probe", list(range(100, 140)))
    num_cached, att = attribution(manager, probe)
    assert num_cached == 0
    assert att.reason == "no_match"
    assert att.block_index == 0
    assert att.extra_key_kinds == ()


def test_miss_attribution_salt_change_reports_key_kind_only():
    """Identical tokens with a different cache_salt must be no_match."""
    manager = make_manager(num_blocks=16)
    tokens = list(range(40))
    warm_prefix(manager, "warm", tokens, cache_salt="A")

    probe = make_request("probe", tokens, cache_salt="B")
    num_cached, att = attribution(manager, probe)
    assert num_cached == 0
    assert att.reason == "no_match"
    assert att.extra_key_kinds == ("cache_salt",)


def test_miss_attribution_mm_extra_key_kinds():
    manager = make_manager(num_blocks=16)
    probe = make_request(
        "probe",
        list(range(40)),
        mm_positions=[PlaceholderRange(offset=0, length=4)],
        mm_hashes=["img"],
    )
    _, att = attribution(manager, probe)
    assert att.reason == "no_match"
    assert "mm" in att.extra_key_kinds


def test_miss_attribution_evicted_after_cache_pressure():
    manager = make_manager(num_blocks=4)
    tokens = list(range(40))
    warm_prefix(manager, "warm", tokens)

    # Allocate an unrelated prompt over every free block, forcing the warm
    # blocks out of cached_block_hash_to_block.
    other = make_request("other", list(range(100, 140)))
    assert manager.allocate_slots(other, len(tokens)) is not None
    manager.free(other)

    probe = make_request("probe", tokens)
    num_cached, att = attribution(manager, probe)
    assert num_cached == 0
    assert att.reason == "evicted"


def test_miss_attribution_disabled_by_default():
    """Turning the diagnostic off must leave cache hits unchanged."""
    manager = make_manager(num_blocks=16, enable_attribution=False)
    assert manager.block_pool.recent_evictions is None
    assert not manager.enable_prefix_cache_miss_attribution
    warm_prefix(manager, "warm", list(range(40)))

    # Two full blocks are resident; the default path must still hit both.
    replay = make_request("replay", list(range(48)))
    _, num_cached, _ = manager.get_computed_blocks(replay)
    assert num_cached == 32
    assert replay.prefix_cache_miss_attributions == ()


def test_recent_evictions_is_bounded_lru():
    history = RecentEvictions(max_size=2)
    h0 = make_block_hash_with_group_id(BlockHash(b"a"), 0)
    h1 = make_block_hash_with_group_id(BlockHash(b"b"), 0)
    h2 = make_block_hash_with_group_id(BlockHash(b"c"), 0)
    history.add([h0, h1])
    history.add([h0])
    history.add([h2])
    assert len(history) == 2
    assert h0 in history
    assert h2 in history
    assert h1 not in history
