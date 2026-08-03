# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def scheduler():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config)


@pytest.fixture
def hybrid_scheduler():
    """Scheduler for a hybrid model: full attention plus a Mamba group."""
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size,
        mamba_enabled=True,
        num_blocks=10000,
    )
    return create_scheduler(vllm_config, kv_cache_config=kv_cache_config)


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_async_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    (req2_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request2.request_id)
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request1.request_id, request3.request_id},
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    min_invalid_block_idx = min(invalid_block_idxs)

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        if request.request_id == request2.request_id:
            assert request.num_computed_tokens == (
                min_invalid_block_idx * scheduler.block_size
            )
        else:
            assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.failed_recving_kv_req_ids == {request2.request_id}
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_sync_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    assert len(scheduler.running) == 3
    assert len(scheduler_output.scheduled_new_reqs) == 3
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    req2_block_ids = scheduler_output.scheduled_new_reqs[1].block_ids[0]
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2, request3],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == request2.request_id
    assert scheduler.running[0].num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size
    )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3
    assert scheduler.connector.request_finished.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "num_common_prefix_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, 50, {0, 49}),
        (100, 99, 50, {25, 49}),
        (100, 99, 50, {49}),
    ],
)
def test_sync_load_failure_with_shared_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    num_common_prefix_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks >= num_common_prefix_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request1)
    request2 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request2)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 2
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2

    # Simulate a failure in loading some of the shared blocks.
    req1_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req1_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2], invalid_block_ids=invalid_block_ids, use_eos=True
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # req_id -> num_computed_tokens
    # all the common prefix blocks will be computed by request1
    expected_computed_tokens = {
        request1.request_id: min(invalid_block_idxs) * scheduler.block_size,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    for request in scheduler.running:
        assert (
            request.num_computed_tokens == expected_computed_tokens[request.request_id]
        )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 50, 98}),
        (100, 99, {98, 50, 0}),
    ],
)
def test_async_progressive_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1
    assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
    assert request.num_computed_tokens == num_external_computed_tokens
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    min_invalid_block_idx = max(invalid_block_idxs) + 1
    # Simulate failures when progressively loading request blocks.
    for invalid_block_idx in invalid_block_idxs:
        (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        invalid_block_ids = {req_block_ids[invalid_block_idx]}
        model_runner_output = create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        )

        scheduler.update_from_output(scheduler_output, model_runner_output)

        min_invalid_block_idx = min(min_invalid_block_idx, invalid_block_idx)

        assert len(scheduler.waiting) == 0
        assert len(scheduler.skipped_waiting) == 1
        assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
        assert request.num_computed_tokens == (
            min_invalid_block_idx * scheduler.block_size
        )
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        assert scheduler.connector.get_num_new_matched_tokens.call_count == 1


def _schedule_hybrid_async_load(
    scheduler: Scheduler, num_prompt_blocks: int, num_external_computed_blocks: int
) -> Request:
    """Put one request into an async external load on a hybrid model."""
    request = create_request(num_tokens=num_prompt_blocks * scheduler.block_size)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_blocks * scheduler.block_size},
            async_load=True,
        )
    )
    scheduler.connector.take_events.return_value = ()
    return request


def _report_invalid_blocks(
    scheduler: Scheduler, scheduler_output, invalid_block_ids: set[int]
) -> None:
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        ),
    )


@pytest.mark.parametrize("failed_group_idx", [0, 1])
def test_hybrid_load_failure_recomputes_whole_request(
    hybrid_scheduler: Scheduler, failed_group_idx: int
):
    """A hybrid model must recover, not crash.

    Its groups sit on different block grids, so there is no single truncation
    point; the request restarts from scratch instead.
    """
    num_prompt_blocks, num_external_computed_blocks = 10, 9
    request = _schedule_hybrid_async_load(
        hybrid_scheduler, num_prompt_blocks, num_external_computed_blocks
    )
    scheduler_output = hybrid_scheduler.schedule()

    block_ids_per_group = hybrid_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert len(block_ids_per_group) == 2, "expected a hybrid (multi-group) layout"
    null_block_id = hybrid_scheduler.kv_cache_manager.block_pool.null_block.block_id
    # The Mamba group backs only its aligned snapshot with a real block, so
    # pick a block that actually belongs to this request.
    failed_block = next(
        block_id
        for block_id in block_ids_per_group[failed_group_idx]
        if block_id != null_block_id
    )

    _report_invalid_blocks(hybrid_scheduler, scheduler_output, {failed_block})

    assert request.num_computed_tokens == 0
    assert hybrid_scheduler.failed_recving_kv_req_ids == {request.request_id}
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_hybrid_load_failure_ignores_null_block(hybrid_scheduler: Scheduler):
    """The null block is shared by every request and holds no KV, so reporting
    it must not restart anyone."""
    num_prompt_blocks, num_external_computed_blocks = 10, 9
    request = _schedule_hybrid_async_load(
        hybrid_scheduler, num_prompt_blocks, num_external_computed_blocks
    )
    scheduler_output = hybrid_scheduler.schedule()

    null_block_id = hybrid_scheduler.kv_cache_manager.block_pool.null_block.block_id
    _, mamba_block_ids = hybrid_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    # The Mamba group parks its unaligned positions on the null block, so a
    # request that never failed would otherwise be restarted by this report.
    assert mamba_block_ids[0] == null_block_id

    num_computed_tokens = request.num_computed_tokens
    _report_invalid_blocks(hybrid_scheduler, scheduler_output, {null_block_id})

    assert request.num_computed_tokens == num_computed_tokens
    assert not hybrid_scheduler.failed_recving_kv_req_ids


def _make_deepseek_v4_five_group_config(num_blocks: int) -> KVCacheConfig:
    """Build the scheduler-side projection of V4-Flash's packed KV layout."""

    def mla_spec() -> MLAAttentionSpec:
        return MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
            model_version="deepseek_v4",
        )

    def swa_spec(
        *, block_size: int, sliding_window: int, head_size: int
    ) -> SlidingWindowMLASpec:
        return SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.float32,
            sliding_window=sliding_window,
            model_version="deepseek_v4",
        )

    # DeepSeek-V4 packs these into five physical groups. The two SWA groups
    # have identical scheduling semantics but are split to align layer tuples.
    groups = [
        KVCacheGroupSpec(["full_mla_and_indexer"], mla_spec()),
        KVCacheGroupSpec(
            ["c4_state"],
            swa_spec(block_size=4, sliding_window=8, head_size=2048),
        ),
        KVCacheGroupSpec(
            ["c128_state"],
            swa_spec(block_size=8, sliding_window=128, head_size=2048),
        ),
        KVCacheGroupSpec(
            ["swa.0"],
            swa_spec(block_size=64, sliding_window=512, head_size=512),
        ),
        KVCacheGroupSpec(
            ["swa.1"],
            swa_spec(block_size=64, sliding_window=512, head_size=512),
        ),
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _create_deepseek_v4_scheduler(
    kv_load_failure_policy: Literal["recompute", "fail"] = "recompute",
) -> Scheduler:
    num_blocks = 10000
    vllm_config = create_vllm_config(
        block_size=256,
        max_num_batched_tokens=4096,
        max_model_len=8192,
        kv_load_failure_policy=kv_load_failure_policy,
    )
    return create_scheduler(
        vllm_config,
        num_blocks=num_blocks,
        kv_cache_config=_make_deepseek_v4_five_group_config(num_blocks),
        hash_block_size=4,
    )


@pytest.fixture
def deepseek_v4_scheduler() -> Scheduler:
    return _create_deepseek_v4_scheduler()


def _schedule_deepseek_v4_async_load(
    scheduler: Scheduler,
) -> tuple[Request, SchedulerOutput]:
    num_external_tokens = 8 * scheduler.block_size
    request = create_request(num_tokens=9 * scheduler.block_size, block_size=4)
    scheduler.add_request(request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_tokens}, async_load=True
        )
    )
    scheduler.connector.take_events.return_value = ()
    return request, scheduler.schedule()


@pytest.mark.parametrize("failed_group_idx", range(5))
def test_deepseek_v4_load_failure_recomputes_whole_request(
    deepseek_v4_scheduler: Scheduler, failed_group_idx: int
):
    """A failed load in any V4-Flash KV group must not kill EngineCore."""
    request, scheduler_output = _schedule_deepseek_v4_async_load(deepseek_v4_scheduler)

    block_ids_per_group = deepseek_v4_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert len(block_ids_per_group) == 5
    null_block_id = (
        deepseek_v4_scheduler.kv_cache_manager.block_pool.null_block.block_id
    )
    failed_block_id = next(
        block_id
        for block_id in block_ids_per_group[failed_group_idx]
        if block_id != null_block_id
    )

    deepseek_v4_scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids={failed_block_id},
            use_eos=True,
        ),
    )

    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert deepseek_v4_scheduler.failed_recving_kv_req_ids == {request.request_id}


def test_deepseek_v4_load_failure_ignores_shared_null_block(
    deepseek_v4_scheduler: Scheduler,
):
    """The shared null block contains no request KV and cannot fail a load."""
    request, scheduler_output = _schedule_deepseek_v4_async_load(deepseek_v4_scheduler)
    null_block_id = (
        deepseek_v4_scheduler.kv_cache_manager.block_pool.null_block.block_id
    )
    block_ids_per_group = deepseek_v4_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert any(null_block_id in group for group in block_ids_per_group[1:])
    num_computed_tokens = request.num_computed_tokens

    deepseek_v4_scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids={null_block_id},
            use_eos=True,
        ),
    )

    assert request.num_computed_tokens == num_computed_tokens
    assert not deepseek_v4_scheduler.failed_recving_kv_req_ids


def test_deepseek_v4_sync_load_failure_fails_and_evicts_real_blocks():
    """Fail policy must terminate cleanly and never evict the null block."""
    scheduler = _create_deepseek_v4_scheduler(kv_load_failure_policy="fail")
    num_external_tokens = 8 * scheduler.block_size
    request = create_request(num_tokens=9 * scheduler.block_size, block_size=4)
    scheduler.add_request(request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_tokens}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler_output = scheduler.schedule()

    block_ids_per_group = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(block_ids_per_group) == 5
    null_block_id = scheduler.kv_cache_manager.block_pool.null_block.block_id
    real_block_ids = {
        block_id
        for group_block_ids in block_ids_per_group
        for block_id in group_block_ids
        if block_id != null_block_id
    }
    failed_block_id = next(
        block_id for block_id in block_ids_per_group[2] if block_id != null_block_id
    )

    evicted_block_ids: list[set[int]] = []
    original_evict_blocks = scheduler.kv_cache_manager.evict_blocks

    def evict_blocks_spy(block_ids: set[int]) -> None:
        evicted_block_ids.append(set(block_ids))
        original_evict_blocks(block_ids)

    with (
        patch.object(
            scheduler.kv_cache_manager,
            "evict_blocks",
            side_effect=evict_blocks_spy,
        ),
        patch.object(scheduler, "_connector_finished", return_value=(False, None)),
    ):
        outputs = scheduler.update_from_output(
            scheduler_output,
            create_model_runner_output(
                [request],
                invalid_block_ids={failed_block_id},
                use_eos=True,
            ),
        )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR
    assert request.request_id not in scheduler.requests
    assert evicted_block_ids == [real_block_ids]
    assert null_block_id not in evicted_block_ids[0]

    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    assert engine_outputs.outputs[0].request_id == request.request_id
    assert engine_outputs.outputs[0].finish_reason == FinishReason.ERROR
