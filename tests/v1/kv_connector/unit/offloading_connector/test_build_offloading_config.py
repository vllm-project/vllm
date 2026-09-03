# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the hash-divisibility assert in build_offloading_config().

Hybrid models can carry a KV cache group that opts out of prefix caching
(e.g. CircularBufferSpec, a one-block-per-request ring whose block size is
unrelated to the hash granularity). Such a group is never addressed by
block hashes, so constraining its block size by tokens_per_hash makes
native offloading unbootable on those models.
"""

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _full_attention_spec,
    _make_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# The attention group sets the hash granularity; the ring group's block is
# far smaller and unrelated to it.
ATTENTION_BLOCK_SIZE = 1152
RING_BLOCK_SIZE = 4


def _ring_spec(block_size: int = RING_BLOCK_SIZE) -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
    )


def _kv_cache_config(*specs) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def test_non_prefix_cacheable_group_is_exempt():
    """A ring group whose block size does not divide the hash unit boots.

    Without the exemption this raises AssertionError, and no hybrid model
    carrying such a group can start with native offloading at all.
    """
    config = build_offloading_config(
        _make_vllm_config(),
        _kv_cache_config(
            _full_attention_spec(block_size=ATTENTION_BLOCK_SIZE), _ring_spec()
        ),
    )

    # The hash unit comes from the prefix-cacheable group alone, so the ring
    # block size (4) never has to divide it.
    assert config.cache.tokens_per_hash == ATTENTION_BLOCK_SIZE
    assert [group.tokens_per_block for group in config.groups] == [
        ATTENTION_BLOCK_SIZE,
        RING_BLOCK_SIZE,
    ]


def test_divisibility_still_enforced_for_prefix_cacheable_groups():
    """Prefix-cacheable groups keep the assert.

    A mamba group outside "align" mode pins the hash unit to the LCM of the
    group block sizes (48), which the prefix-cacheable attention group's own
    block size (16) does not divide.
    """
    kv_cache_config = _kv_cache_config(
        _full_attention_spec(block_size=16),
        MambaSpec(
            block_size=24,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="none",
        ),
    )

    with pytest.raises(AssertionError, match="not divisible"):
        build_offloading_config(_make_vllm_config(), kv_cache_config)


def test_all_non_prefix_cacheable_falls_back_to_every_group():
    """With nothing prefix-cacheable, every group is asserted again.

    Mirrors the ``or group_block_sizes`` fallback that
    resolve_kv_cache_block_sizes() uses when deriving the hash unit, so the
    assert degenerates into checking every group rather than checking none.
    """
    config = build_offloading_config(
        _make_vllm_config(),
        _kv_cache_config(_ring_spec(4), _ring_spec(8)),
    )

    # GCD of the ring block sizes, since no group participates in hashing.
    assert config.cache.tokens_per_hash == 4
