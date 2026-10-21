# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests


def _create_hybrid_transition_scheduler(
    tmp_path, num_requests=7, usable_blocks=148, checkpoint_blocks=0, max_num_seqs=None
):
    """Construct initialized hybrid decodes using native allocation APIs."""
    import torch
    from transformers import OPTConfig

    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    # Only configuration is needed; no model weights or serving engine.
    OPTConfig(
        architectures=["OPTForCausalLM"],
        max_position_embeddings=65536,
        hidden_size=64,
        word_embed_proj_dim=64,
        ffn_dim=128,
        num_attention_heads=1,
        num_hidden_layers=5,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        skip_tokenizer_init=True,
        max_model_len=65536,
    )
    parallel_config = ParallelConfig()
    config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=num_requests if max_num_seqs is None else max_num_seqs,
            max_num_batched_tokens=192,
            max_model_len=65536,
            enable_chunked_prefill=True,
            async_scheduling=True,
            watermark=0.0,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=8704,
            enable_prefix_caching=False,
            mamba_cache_mode="align",
        ),
        speculative_config=SpeculativeConfig(
            model=str(tmp_path),
            method="draft_model",
            num_speculative_tokens=5,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        ),
    )
    attention = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    groups = [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=8704, **attention)
        ),
        KVCacheGroupSpec(["ring"], CircularBufferSpec(block_size=16, **attention)),
        *[
            KVCacheGroupSpec(
                [f"mamba_{i}"],
                MambaSpec(
                    block_size=8704,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=5,
                    num_prefill_checkpoint_blocks=checkpoint_blocks,
                ),
            )
            for i in range(3)
        ],
    ]
    kv_config = KVCacheConfig(
        num_blocks=usable_blocks + 1, kv_cache_tensors=[], kv_cache_groups=groups
    )
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    register_all_kvcache_specs(config)
    scheduler = AsyncScheduler(
        vllm_config=config,
        kv_cache_config=kv_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=8704,
        hash_block_size=128,
        log_stats=False,
    )
    assert scheduler.num_lookahead_tokens == 5
    requests = create_requests(
        num_requests=num_requests,
        num_tokens=8499,
        max_tokens=1024,
        ignore_eos=True,
        block_size=128,
    )
    for request in requests:
        scheduler.add_request(request)
        assert (
            scheduler.kv_cache_manager.allocate_slots(
                request,
                0,
                num_external_computed_tokens=8498,
                delay_cache_blocks=True,
            )
            is not None
        )
        scheduler.waiting.remove_requests([request])
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids([7] * 196)
        request.num_computed_tokens = 8694
        request.spec_token_ids = [7] * 5
        request.is_prefill_chunk = False
        scheduler.running.append(request)
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == usable_blocks - 20 * num_requests
    return scheduler, requests


def _complete_hybrid_step(scheduler, output, reject_drafts=False, prefill_ids=()):
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[
                []
                if rid in prefill_ids
                else [7]
                * (
                    1
                    if reject_drafts
                    else 1 + len(output.scheduled_spec_decode_tokens.get(rid, []))
                )
                for rid in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


@pytest.mark.parametrize("reject_drafts", [False, True])
def test_hybrid_transition_headroom_completes_existing_boundary_case(
    tmp_path, reject_drafts
):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    for _ in range(6000):
        if not scheduler.running:
            break
        output = scheduler.schedule()
        assert not output.preempted_req_ids
        assert output.total_num_scheduled_tokens > 0
        _complete_hybrid_step(scheduler, output, reject_drafts)
    assert not scheduler.running and not scheduler.waiting
    assert [r.num_output_tokens for r in requests] == [1024] * 7
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 148


def test_hybrid_transition_headroom_does_not_allocate_or_move_state_early(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    cache = scheduler.kv_cache_manager
    states = [
        list(m.req_to_blocks[requests[0].request_id])
        for m in cache.coordinator.single_type_managers[2:]
    ]
    assert scheduler._transition_blocks(requests[0]) == 4
    assert cache.block_pool.get_num_free_blocks() == 8
    output = scheduler.schedule()
    assert 0 < len(output.num_scheduled_tokens) < 7
    assert cache.block_pool.get_num_free_blocks() == 8 - len(
        output.num_scheduled_tokens
    )
    assert [
        list(m.req_to_blocks[requests[0].request_id])
        for m in cache.coordinator.single_type_managers[2:]
    ] == states
    assert scheduler._transition_blocks(requests[0]) == 3


def test_hybrid_transition_true_shortage_keeps_existing_preemption(tmp_path):
    scheduler, _ = _create_hybrid_transition_scheduler(tmp_path, 2, 43)
    output = scheduler.schedule()
    assert output.preempted_req_ids
    assert output.total_num_scheduled_tokens > 0
    _complete_hybrid_step(scheduler, output)


def test_hybrid_transition_does_not_delay_higher_priority_for_scheduled_work(tmp_path):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    for request in requests:
        request.priority = 20
    requests[0].priority = 10
    requests[1].priority = 0
    first = scheduler.schedule()
    assert not first.preempted_req_ids
    _complete_hybrid_step(scheduler, first)
    scheduler.policy = SchedulingPolicy.PRIORITY
    second = scheduler.schedule()
    assert requests[1].request_id in second.num_scheduled_tokens
    assert second.preempted_req_ids
    assert requests[1].request_id not in second.preempted_req_ids
    _complete_hybrid_step(scheduler, second)


def test_hybrid_transition_cancel_respects_inflight_and_shared_references(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    scheduler.defer_block_free = True
    cache = scheduler.kv_cache_manager
    output = scheduler.schedule()
    request = requests[0]
    held = [
        m.req_to_blocks[request.request_id][0]
        for m in cache.coordinator.single_type_managers[2:]
    ]
    cache.block_pool.touch(held)
    before = cache.block_pool.get_num_free_blocks()
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert cache.block_pool.get_num_free_blocks() == before
    _complete_hybrid_step(scheduler, output)
    assert [b.ref_cnt for b in held] == [1, 1, 1]
    cache.block_pool.free_blocks(held)
    for other in list(scheduler.requests.values()):
        scheduler.finish_requests(other.request_id, RequestStatus.FINISHED_ABORTED)
    assert cache.block_pool.get_num_free_blocks() == 148


def test_hybrid_transition_query_is_noop_for_checkpoint_layout(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(
        tmp_path, 1, 80, checkpoint_blocks=1
    )
    cache = scheduler.kv_cache_manager
    managers = cache.coordinator.single_type_managers[2:]
    before = [dict(m._checkpoints) for m in managers]
    assert scheduler._running_transition_headroom() is None
    assert [dict(m._checkpoints) for m in managers] == before


def test_hybrid_transition_final_token_needs_no_future_headroom(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path, 2, 43)
    for request in requests:
        request.max_tokens = request.num_output_tokens + 1
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {r.request_id for r in requests}
    assert not output.preempted_req_ids
    _complete_hybrid_step(scheduler, output, reject_drafts=True)
    assert not scheduler.running
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 43


def test_hybrid_transition_mixed_priorities_keep_preemption(tmp_path):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    first = scheduler.schedule()
    _complete_hybrid_step(scheduler, first)
    scheduler.policy = SchedulingPolicy.PRIORITY
    for request in requests:
        request.priority = 20
    requests[0].priority = 0
    requests[1].priority = 10
    requests[2].priority = 5
    second = scheduler.schedule()
    assert requests[2].request_id in second.num_scheduled_tokens
    assert second.preempted_req_ids
    assert requests[0].request_id not in second.preempted_req_ids
    assert requests[2].request_id not in second.preempted_req_ids
    _complete_hybrid_step(scheduler, second)


def test_hybrid_transition_one_output_token_can_still_have_prefill_work(tmp_path):
    scheduler, old_requests = _create_hybrid_transition_scheduler(tmp_path, 1, 23)
    scheduler.finish_requests(
        old_requests[0].request_id, RequestStatus.FINISHED_ABORTED
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=20000, max_tokens=1, req_ids=["prefill"]
    )
    scheduler.add_request(request)
    cache = scheduler.kv_cache_manager
    assert (
        cache.allocate_slots(
            request, 0, num_external_computed_tokens=8694, delay_cache_blocks=True
        )
        is not None
    )
    request.num_computed_tokens = 8694
    scheduler.waiting.remove_requests([request])
    request.status = RequestStatus.RUNNING
    request.is_prefill_chunk = True
    scheduler.running.append(request)
    assert cache.block_pool.get_num_free_blocks() == 3
    assert (
        cache.allocate_slots(
            request,
            6,
            num_lookahead_tokens=5,
            transition_blocks=scheduler._transition_blocks(request),
        )
        is None
    )
    assert cache.block_pool.get_num_free_blocks() == 3


def test_hybrid_transition_headroom_respects_target_context_limit(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path, 2, 43)
    cache = scheduler.kv_cache_manager
    scheduler.max_model_len = cache.max_model_len = 8705
    # Attention may cover position 8704 for drafting, but target execution
    # cannot advance beyond 8704 inputs under this context limit.
    for request in requests:
        assert cache.allocate_slots(request, 6, num_lookahead_tokens=5) is not None
    assert cache.block_pool.get_num_free_blocks() == 1


def _waiting_transition_request(scheduler, name="waiting-transition"):
    request = create_requests(
        num_requests=1,
        num_tokens=8499,
        max_tokens=1024,
        ignore_eos=True,
        block_size=128,
    )[0]
    request.request_id = name
    scheduler.add_request(request)
    return request


def test_transition_budget_survives_waiting_admission(tmp_path):
    scheduler, (request,) = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=1, usable_blocks=41, max_num_seqs=2
    )
    waiting = _waiting_transition_request(scheduler)
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert waiting.request_id not in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 20
    _complete_hybrid_step(scheduler, output)
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 17


def test_blocked_head_does_not_stop_zero_allocation_work_or_admit_waiters(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=2, usable_blocks=44, max_num_seqs=3
    )
    assert (
        scheduler.kv_cache_manager.allocate_slots(
            requests[1], 6, num_lookahead_tokens=5
        )
        is not None
    )
    waiting = _waiting_transition_request(scheduler)
    output = scheduler.schedule()
    assert requests[0].request_id not in output.num_scheduled_tokens
    assert requests[1].request_id in output.num_scheduled_tokens
    assert waiting.request_id not in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 3


def test_running_cow_cannot_spend_another_request_transition_budget(tmp_path):
    scheduler, (request,) = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=1, usable_blocks=44, max_num_seqs=2
    )
    other = _waiting_transition_request(scheduler, "running-cow")
    cache = scheduler.kv_cache_manager
    assert (
        cache.allocate_slots(
            other, 0, num_external_computed_tokens=8448, delay_cache_blocks=True
        )
        is not None
    )
    scheduler.waiting.remove_requests([other])
    other.status = RequestStatus.RUNNING
    other.num_computed_tokens = 8448
    other.is_prefill_chunk = True
    scheduler.running.append(other)
    for manager in cache.coordinator.single_type_managers[2:]:
        assert manager._cache_partial_tail_block(other, 8448, 0) is not None
    assert cache.allocate_slots(request, 6, num_lookahead_tokens=5) is not None
    assert cache.block_pool.get_num_free_blocks() == 3
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert other.request_id not in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert cache.block_pool.get_num_free_blocks() == 3
    assert all(
        other.request_id in m._partial_hit_reqs
        for m in cache.coordinator.single_type_managers[2:]
    )


def test_cancelled_transition_does_not_leave_admission_budget(tmp_path):
    scheduler, (request,) = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=1, usable_blocks=41, max_num_seqs=2
    )
    waiting = _waiting_transition_request(scheduler)
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert waiting.request_id in output.num_scheduled_tokens
    assert request.request_id not in output.num_scheduled_tokens
    assert not hasattr(scheduler.kv_cache_manager, "forward_reservation")


def test_transition_scan_never_releases_shared_or_inflight_blocks(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    pool = scheduler.kv_cache_manager.block_pool
    block = scheduler.kv_cache_manager.coordinator.single_type_managers[
        2
    ].req_to_blocks[requests[0].request_id][0]
    pool.touch([block])
    before = [(b.block_id, b.ref_cnt) for b in pool.blocks]
    needs = scheduler._running_transition_headroom()
    assert needs and needs[requests[0]] > 0
    assert before == [(b.block_id, b.ref_cnt) for b in pool.blocks]
    pool.free_blocks([block])


def test_transition_does_not_reserve_for_preempted_cached_entry(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    needs = scheduler._running_transition_headroom()
    for request in requests:
        request.status = RequestStatus.PREEMPTED
    assert scheduler._transition_reserve(needs) == 0


def test_waiting_request_is_eventually_admitted_after_transition_space_available(
    tmp_path,
):
    scheduler, (request,) = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=1, usable_blocks=41, max_num_seqs=2
    )
    waiting = _waiting_transition_request(scheduler)
    for _ in range(1024):
        output = scheduler.schedule()
        assert not output.preempted_req_ids
        if waiting.request_id in output.num_scheduled_tokens:
            assert request.is_finished() or scheduler._transition_blocks(request) == 0
            break
        assert output.total_num_scheduled_tokens > 0
        _complete_hybrid_step(scheduler, output)
    else:
        pytest.fail("waiting admission did not resume after finite running work")


def test_sufficient_capacity_does_not_block_waiting_admission(tmp_path):
    scheduler, (request,) = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=1, usable_blocks=64, max_num_seqs=2
    )
    waiting = _waiting_transition_request(scheduler)
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert waiting.request_id in output.num_scheduled_tokens
    assert not output.preempted_req_ids


def test_pending_frees_do_not_block_other_zero_allocation_work(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(
        tmp_path, num_requests=2, usable_blocks=44
    )
    assert (
        scheduler.kv_cache_manager.allocate_slots(
            requests[1], 6, num_lookahead_tokens=5
        )
        is not None
    )
    scheduler.connector = Mock()
    scheduler.connector.has_pending_block_frees.return_value = True
    output = scheduler.schedule()
    assert requests[1].request_id in output.num_scheduled_tokens
    assert requests[0].request_id not in output.num_scheduled_tokens
    assert not output.preempted_req_ids


def test_transition_growth_across_multiple_boundaries(tmp_path):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    for request in requests:
        request.max_tokens = request.sampling_params.max_tokens = 20000
    for _ in range(7 * 20000):
        if not scheduler.running:
            break
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        assert not output.preempted_req_ids
        _complete_hybrid_step(scheduler, output)
    else:
        pytest.fail("multiple-boundary work failed to finish")
    assert [r.num_output_tokens for r in requests] == [20000] * 7
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 148


@pytest.mark.parametrize("reject_drafts", [False, True])
def test_transition_budget_with_two_inflight_steps(tmp_path, reject_drafts):
    scheduler, requests = _create_hybrid_transition_scheduler(tmp_path)
    pending: deque[SchedulerOutput] = deque()
    for _ in range(7 * 1024 * 2):
        output = scheduler.schedule()
        assert not output.preempted_req_ids
        if output.total_num_scheduled_tokens:
            pending.append(output)
        if pending and (len(pending) >= 2 or not output.total_num_scheduled_tokens):
            _complete_hybrid_step(scheduler, pending.popleft(), reject_drafts)
        if not scheduler.running and not pending:
            break
        assert pending or output.total_num_scheduled_tokens
    else:
        pytest.fail("two-step pipeline did not finish")
    assert [r.num_output_tokens for r in requests] == [1024] * 7
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 148
