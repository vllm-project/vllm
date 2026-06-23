# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: hybrid (full-attention + Mamba) + KV-connector per-group
prefix-hit divergence.

For hybrid models the scheduler queries each KV-cache group independently
(``find_longest_cache_hit_per_group``). Under block pressure a request's
full-attention (FA) prefix blocks can be evicted while a deeper Mamba state
survives, so the per-group hits *diverge*: ``(FA=short, Mamba=deep)``.

The scheduler must then admit the request with a block set that is consistent at
``num_computed`` for every group. It takes the spine (FA) ``min`` as the local
hit and, once the connector's external length is known, reconciles the
divergence:

- ``ext == 0`` (connector supplies nothing): re-query the convergent hit, which
  re-finds the Mamba state valid at the shorter, all-groups-consistent boundary.
- ``ext > 0`` (connector supplies the prefix incl. the Mamba state up to
  ``num_computed``): trim each group's hit to ``num_computed``; the connector
  reloads the state at that depth.

This test pins that behavior for ``ext`` in {0, 256, 512}. Scheduler-level, no
GPU: it builds a hybrid ``Scheduler`` with a ``MockKVConnector``, seeds a
prefix, surgically evicts the FA tail, then replays the prefix.
"""

import pytest
import torch

import tests.v1.kv_connector.unit.utils  # noqa: F401  (registers MockKVConnector)
from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 256  # scheduler / per-group block size
HASH_BLOCK_SIZE = 4  # mamba "all"-mode hashing granularity
NUM_PREFIX_BLOCKS = 4  # cached prefix: FA[0..3] + mamba state@3
FA_GROUP_ID = 0
MAMBA_GROUP_ID = 1


def _build_hybrid_scheduler(matched_tokens: int, num_blocks: int = 800) -> Scheduler:
    """Hybrid scheduler: group 0 = FullAttention, group 1 = Mamba(mode="all"),
    plus a ``MockKVConnector`` returning ``matched_tokens`` external tokens."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="all",
    )
    cache_config.num_gpu_blocks = num_blocks
    kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "matched_tokens": matched_tokens,
            "is_async": False,
        },
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=1),
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["dense_full_attention"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.uint8,
            ),
        ),
        KVCacheGroupSpec(
            ["mamba_state"],
            MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=((1, 1),),
                dtypes=(torch.uint8,),
                mamba_cache_mode="all",
            ),
        ),
    ]
    register_all_kvcache_specs(vllm_config)
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        ),
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
        log_stats=True,
    )
    assert isinstance(scheduler.kv_cache_manager.coordinator, HybridKVCacheCoordinator)
    return scheduler


def _seed_prefix(scheduler: Scheduler) -> tuple[list[int], list[int]]:
    """Allocate + free a full prefix so both groups are cached. Returns the FA
    and Mamba groups' block ids in sequence order (for surgical eviction)."""
    [fill] = create_requests(
        num_requests=1,
        num_tokens=NUM_PREFIX_BLOCKS * BLOCK_SIZE,
        max_tokens=1,
        same_prompt=True,
        block_size=HASH_BLOCK_SIZE,
        req_ids=["fill"],
    )
    manager = scheduler.kv_cache_manager
    computed_blocks, num_computed = manager.get_computed_blocks(fill)
    assert num_computed == 0
    blocks = manager.allocate_slots(
        fill,
        num_new_tokens=fill.num_tokens,
        num_new_computed_tokens=num_computed,
        new_computed_blocks=computed_blocks,
    )
    assert blocks is not None
    fa_block_ids = [b.block_id for b in blocks.blocks[FA_GROUP_ID]]
    mamba_block_ids = [b.block_id for b in blocks.blocks[MAMBA_GROUP_ID]]
    manager.free(fill)
    return fa_block_ids, mamba_block_ids


@pytest.mark.parametrize(
    "returned_blocks,expected_num_computed",
    [
        # Case A -- connector returns 0 blk (ext=0):
        #   pos:     0    1    2    3   |  connector: 0 blk
        #   Mamba:  m0   --   --   m3   |
        #   FA:     f0   f1   --   --   |
        # No external blocks, so reconcile by re-querying the convergent hit. FA
        # reaches 512, but the state@1 it would need there is evicted, so the
        # convergent boundary drops to 256, where f0 + the resident state@0 agree.
        # Correct: num_computed = 256, served from the local cache (re-query, not
        # a trim -- trimming the sparse [.,.,.,m3] would drop the only state).
        (0, 256),
        # Case B -- connector returns 1 blk (ext=256): num_computed = 512+256 = 768.
        #   pos:     0    1    2    3   |  connector: 1 blk
        #   Mamba:  m0   --   --   m3   |   m2
        #   FA:     f0   f1   --   --   |   f2
        # The connector supplies the prefix (incl. state) up to 768, so reconcile
        # by trimming each group's hit to 768 (3 blk); the deeper state@3 is
        # dropped so external allocation stays non-negative and the connector
        # reloads the state at the 768 boundary. Correct: num_computed = 768.
        (1, 768),
        # Case C -- connector returns 2 blk (ext=512): num_computed = 512+512 = 1024.
        #   pos:     0    1    2    3   |  connector: 2 blk
        #   Mamba:  m0   --   --   m3   |   m3
        #   FA:     f0   f1   --   --   |   f2   f3
        # The connector supplies the prefix up to 1024, so reconcile by trimming
        # each group's hit to 1024 (4 blk); the state@3 lands exactly on the
        # boundary. Correct: num_computed = 1024, consistent across all groups.
        (2, 1024),
    ],
)
def test_hybrid_diverged_prefix_hit(returned_blocks, expected_num_computed):
    """A hybrid request whose per-group hits diverge (FA evicted, Mamba state
    survives) must be admitted with a block set consistent at
    ``num_computed = local + ext`` for every per-group hit length the connector
    reports -- no crash, no over-reported local prefix."""
    ext_tokens = returned_blocks * BLOCK_SIZE
    scheduler = _build_hybrid_scheduler(matched_tokens=ext_tokens)
    coordinator = scheduler.kv_cache_manager.coordinator
    block_pool = scheduler.kv_cache_manager.block_pool

    # Seed a 4-block prefix, then evict the FA tail (FA[2..3]) and the *middle*
    # Mamba states (m1, m2). The connector is still streaming the endpoint states
    # (m0, m3), which keeps them referenced, so under eviction pressure only the
    # unreferenced middle states drop:
    #
    #   HBM resident after eviction (positions are 256-token blocks):
    #       pos:     0    1    2    3
    #       Mamba:  m0   --   --   m3     (m1, m2 evicted; m0, m3 pinned/streaming)
    #       FA:     f0   f1   --   --     (FA[2..3] evicted; f0, f1 survive)
    #
    # => per-group hit diverges: FA reaches 512 (f0,f1), Mamba reaches 1024 (its
    #    hit only needs the deepest state m3, which is still resident).
    fa_block_ids, mamba_block_ids = _seed_prefix(scheduler)
    block_pool.evict_blocks(
        {fa_block_ids[2], fa_block_ids[3], mamba_block_ids[1], mamba_block_ids[2]}
    )

    [replay] = create_requests(
        num_requests=1,
        num_tokens=(NUM_PREFIX_BLOCKS + 1) * BLOCK_SIZE,
        max_tokens=1,
        same_prompt=True,
        block_size=HASH_BLOCK_SIZE,
        req_ids=["replay"],
    )

    # Precondition: the per-group hits really do diverge (FA shorter than Mamba).
    _, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    assert per_group_hits[FA_GROUP_ID] < per_group_hits[MAMBA_GROUP_ID], (
        f"expected diverged hits, got {per_group_hits}"
    )

    scheduler.add_request(replay)
    output = scheduler.schedule()

    num_scheduled = output.num_scheduled_tokens[replay.request_id]
    # num_new_tokens = request.num_tokens - num_computed (uncapped here).
    assert num_scheduled == replay.num_tokens - expected_num_computed

    if ext_tokens == 0:
        # The connector supplies nothing, so the local Mamba block table must
        # itself carry a valid state at the resume boundary. A plain trim of the
        # sparse per-group list (the rejected alternative) would drop it.
        blocks = scheduler.kv_cache_manager.get_blocks(replay.request_id).blocks
        local_mamba = blocks[MAMBA_GROUP_ID][: expected_num_computed // BLOCK_SIZE]
        assert any(
            b is not block_pool.null_block and b.block_hash is not None
            for b in local_mamba
        ), "ext==0 must keep a valid local Mamba state via the convergent re-query"
