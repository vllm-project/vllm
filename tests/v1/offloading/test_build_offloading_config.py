# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for build_offloading_config with non-participating groups.

GLM5Next (KDA+MLA hybrid) carries a KpoolTailSpec group: a 1-block/req
circular scratch buffer that opts out of prefix caching by design. Its
block size (index_kpool=4) is unrelated to the hash granularity
(tokens_per_hash=1152 for the MLA group), so asserting divisibility on
it made native KV offloading unbootable on any hybrid model with such a
tail group.
"""

from types import SimpleNamespace

import pytest

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KpoolTailSpec,
)

# GLM5Next-shaped numbers (index_kpool=4, MLA block 1152, hash unit 1152).
KPOOL_BLOCK_SIZE = 4
MLA_BLOCK_SIZE = 1152
NUM_BLOCKS = 100


def _make_parallel_config():
    return SimpleNamespace(
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        world_size=1,
        data_parallel_index=0,
        data_parallel_size=1,
        data_parallel_rank_local=None,
        distributed_executor_backend="mp",
        nnodes_within_dp=1,
        rank=0,
    )


def _make_vllm_config(model="test-model", dtype="float16"):
    kv_transfer_config = KVTransferConfig()
    kv_transfer_config.kv_connector = "OffloadingConnector"
    kv_transfer_config.kv_role = "kv_both"
    kv_transfer_config.kv_connector_extra_config = {"cpu_bytes_to_use": 1 << 30}
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        parallel_config=_make_parallel_config(),
        model_config=SimpleNamespace(
            model=model,
            dtype=dtype,
            use_mla=False,
            get_total_num_kv_heads=lambda: 1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", prefix_match_unit=None),
        use_v2_model_runner=False,
        kv_events_config=None,
    )


def _make_kv_cache_config(specs):
    groups = [
        KVCacheGroupSpec(layer_names=[f"layer_{i}"], kv_cache_spec=spec)
        for i, spec in enumerate(specs)
    ]
    # One tensor per group; size = num_blocks * block_size * bytes/token(=2).
    tensors = [
        KVCacheTensor(
            name=f"tensor_{i}",
            size=NUM_BLOCKS * spec.block_size * 2,
            block_stride=0,  # unpacked
        )
        for i, spec in enumerate(specs)
    ]
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def test_offloading_boot_with_non_participating_kpool_tail():
    """A hybrid MLA + KpoolTail group set must build an offloading config.

    On stock code this crashes with:
      AssertionError: tokens_per_block=4 not divisible by
      tokens_per_hash=1152
    because the divisibility assert iterated every group, including the
    KpoolTail scratch buffer that opts out of prefix caching.
    """
    specs = [
        FullAttentionSpec(
            block_size=MLA_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=16,
            dtype="float16",
        ),
        KpoolTailSpec(
            block_size=KPOOL_BLOCK_SIZE,
            sliding_window=1024,
            head_size=16,
            dtype="float16",
        ),
    ]
    config = build_offloading_config(_make_vllm_config(), _make_kv_cache_config(specs))
    # The hash granularity is derived from participating groups only
    # (1152 here); the KpoolTail group is present in the offloading
    # groups but exempt from the divisibility constraint.
    assert config.cache.tokens_per_hash == MLA_BLOCK_SIZE
    assert len(config.groups) == 2


def test_offloading_divisibility_still_enforced_for_participating_groups():
    """Participating groups must still satisfy the divisibility assert."""
    specs = [
        FullAttentionSpec(
            block_size=MLA_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=16,
            dtype="float16",
        ),
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype="float16",
        ),
    ]
    with pytest.raises(AssertionError, match="not divisible"):
        build_offloading_config(_make_vllm_config(), _make_kv_cache_config(specs))


def test_offloading_all_non_participating_falls_back():
    """If nothing participates, fall back to asserting on all groups."""
    specs = [
        KpoolTailSpec(
            block_size=KPOOL_BLOCK_SIZE,
            sliding_window=1024,
            head_size=16,
            dtype="float16",
        ),
    ]
    # hash granularity falls back to all group block sizes (4); 4 % 4 == 0
    # so this builds fine — the fallback path must not raise.
    config = build_offloading_config(_make_vllm_config(), _make_kv_cache_config(specs))
    assert config.cache.tokens_per_hash == KPOOL_BLOCK_SIZE
