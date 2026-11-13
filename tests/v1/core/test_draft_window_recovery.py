# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.core.utils import (
    create_requests,
    create_scheduler,
    new_kv_cache_spec,
    new_sliding_window_spec,
)
from vllm.v1.core.draft_window_recovery import plan_draft_window_recovery
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.model_states.default import (
    _protect_draft_recovery_main_slots_cpu,
)


def test_recovery_replays_continuously_from_first_hole():
    plan = plan_draft_window_recovery(
        request_id="request-0",
        main_cached_end=1024,
        retained_history_tokens=(511, 255),
        block_sizes=(64, 64),
        draft_ready_ranges=(
            ((512, 640), (704, 1024)),
            ((768, 896), (960, 1024)),
        ),
    )

    assert plan.required_ranges == ((512, 1024), (768, 1024))
    assert plan.replay_ranges == ((640, 1024),)


def test_recovery_aligns_retained_history_to_block_boundary():
    plan = plan_draft_window_recovery(
        request_id="request-0",
        main_cached_end=1000,
        retained_history_tokens=(63,),
        block_sizes=(4,),
        draft_ready_ranges=((),),
    )

    assert plan.required_ranges == ((936, 1000),)
    assert plan.replay_ranges == ((936, 1000),)


def test_recovery_window_is_anchored_at_main_hit_before_uncached_suffix():
    plan = plan_draft_window_recovery(
        request_id="request-0",
        main_cached_end=1024,
        retained_history_tokens=(511,),
        block_sizes=(64,),
        draft_ready_ranges=((),),
    )

    assert plan.required_ranges == ((512, 1024),)
    assert plan.replay_ranges == ((512, 1024),)


def test_recovery_protects_main_slots_but_keeps_draft_writes():
    slot_mappings = torch.arange(16, dtype=torch.int64)
    slot_mappings = slot_mappings.reshape(2, 8)
    positions = torch.arange(56, 64)

    _protect_draft_recovery_main_slots_cpu(
        slot_mappings=slot_mappings,
        query_start_loc=(0, 8),
        positions=positions,
        main_cached_ends=(60,),
        main_group_ids=(0,),
        pad_slot_id=-1,
    )

    assert slot_mappings[0].tolist() == [-1, -1, -1, -1, 4, 5, 6, 7]
    assert slot_mappings[1].tolist() == list(range(8, 16))


def test_short_suffix_recovers_window_without_discarding_main_hit():
    block_size = 4
    base = create_scheduler(
        block_size=block_size,
        enable_prefix_caching=True,
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=128,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        num_blocks_by_pool=[128, 17],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target"],
                new_kv_cache_spec(block_size=block_size),
                block_pool_id=0,
            ),
            KVCacheGroupSpec(
                ["draft"],
                new_sliding_window_spec(block_size=block_size, sliding_window=16),
                block_pool_id=1,
                role=KVCacheGroupRole.DRAFT_WINDOW,
            ),
        ],
    )
    scheduler = Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )
    manager = scheduler.kv_cache_manager
    (producer,) = create_requests(
        num_requests=1,
        num_tokens=64,
        same_prompt=True,
        block_size=block_size,
        req_ids=["producer"],
    )
    (consumer,) = create_requests(
        num_requests=1,
        num_tokens=48,
        same_prompt=True,
        block_size=block_size,
        req_ids=["consumer"],
    )
    _, churn = create_requests(
        num_requests=2,
        num_tokens=48,
        same_prompt=False,
        block_size=block_size,
        req_ids=["unused", "churn"],
    )
    assert manager.allocate_slots(producer, producer.num_tokens) is not None
    producer.num_computed_tokens = producer.num_tokens
    manager.remove_skipped_blocks(
        producer.request_id,
        producer.num_computed_tokens,
        num_prompt_tokens=producer.num_prompt_tokens,
    )
    manager.cache_blocks(producer, producer.num_computed_tokens)
    manager.free(producer)

    # Fill most of the bounded Draft pool with a different prefix. This evicts
    # the producer's old Draft pages while its latest window remains cached.
    assert manager.allocate_slots(churn, churn.num_tokens) is not None
    churn.num_computed_tokens = churn.num_tokens
    manager.remove_skipped_blocks(
        churn.request_id,
        churn.num_computed_tokens,
        num_prompt_tokens=churn.num_prompt_tokens,
    )
    manager.cache_blocks(churn, churn.num_computed_tokens)
    manager.free(churn)

    scheduler.add_request(consumer)
    output = scheduler.schedule()
    new_request = next(
        request
        for request in output.scheduled_new_reqs
        if request.req_id == consumer.request_id
    )

    # The producer retained only Draft blocks [48, 64). The shorter consumer's
    # reusable Main prefix ends at 44 and needs Draft history from block 28.
    assert new_request.num_computed_tokens == 28
    assert new_request.replay_start == 28
    assert new_request.draft_recovery_main_cached_end == 44
    assert output.num_scheduled_tokens[consumer.request_id] == 8

    draft_blocks = manager.get_blocks(consumer.request_id).blocks[1]
    assert all(block.block_hash is not None for block in draft_blocks[7:9])
    assert all(block.block_hash is None for block in draft_blocks[9:11])
