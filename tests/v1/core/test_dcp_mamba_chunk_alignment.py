# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def test_mamba_align_chunk_split_uses_dcp_effective_block_size():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=1000,
        num_tokens=1000,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=16,
        dcp_world_size=3,
        use_eagle=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=128,
    )

    assert adjusted == 96
    effective_block_size = scheduler.cache_config.block_size * scheduler.dcp_world_size
    assert adjusted % effective_block_size == 0


def test_mamba_align_split_keeps_small_chunks_when_dcp_alignment_exceeds_budget():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=835,
        num_tokens=835,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=208,
        dcp_world_size=3,
        use_eagle=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=256,
    )

    assert adjusted == 256


def test_hybrid_dcp_resolves_global_scheduler_block_and_fine_hash_block():
    torch = pytest.importorskip("torch")
    block_size = 16
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=True,
            hash_block_size=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=3,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )

    assert scheduler_block_size == block_size * 3
    assert hash_block_size == block_size
