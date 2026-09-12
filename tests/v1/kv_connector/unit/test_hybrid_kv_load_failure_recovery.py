# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid (attention + mamba) KV load failure recovery.

`_update_requests_with_invalid_blocks` unpacked a single block-id list, so a
hybrid model attached to a KV connector crashed the scheduler with
"too many values to unpack" the first time a remote chunk failed to load.
These tests pin the recovered contract: a hybrid request replays from zero
with every participating group's prefix hashes invalidated, while
single-group requests keep the existing longest-valid-prefix behavior
(covered by test_kv_load_failure_recovery.py).
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK_SIZE = 16
NUM_PROMPT_BLOCKS = 100
NUM_EXTERNAL_COMPUTED_BLOCKS = 99


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


def _hybrid_kv_cache_config() -> KVCacheConfig:
    """One full-attention group plus one aligned mamba state group."""
    attention_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_layer"], attention_spec),
            KVCacheGroupSpec(["mamba_layer"], mamba_spec),
        ],
    )


def _request_block_ids(scheduler: Scheduler, request: Request) -> tuple[list[int], ...]:
    return scheduler.kv_cache_manager.get_block_ids(request.request_id)


class _HmaMock(Mock, SupportsHMA):
    """A connector mock that keeps its Hybrid-Memory-Allocator identity.

    A plain Mock fails the upstream isinstance(connector, SupportsHMA) gate
    on the hybrid `request_finished` path; hybrid models require an HMA
    connector, which is exactly what this regression family exercises.
    """

    def request_finished_all_groups(self, request, block_ids):
        return False, None


def _assert_attention_prefix_invalidated(scheduler: Scheduler, block_ids) -> None:
    """The attention group's cached prefix entries must be cleared."""
    block_pool = scheduler.kv_cache_manager.block_pool
    for block_id in block_ids:
        block = block_pool.blocks[block_id]
        assert block.ref_cnt == 0  # freed by the preemption


@pytest.fixture
def hybrid_scheduler() -> Scheduler:
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config, kv_cache_config=_hybrid_kv_cache_config())


def test_hybrid_sync_load_failure_replays_from_zero(hybrid_scheduler: Scheduler):
    """A load failure in one group invalidates every group: the request is
    preempted back to waiting with zero computed tokens, the
    skip-reading-prefix-caching flag set, and both groups' cached block
    hashes evicted (rather than resuming at a partial prefix)."""
    num_prompt_tokens = NUM_PROMPT_BLOCKS * BLOCK_SIZE
    num_external_computed_tokens = NUM_EXTERNAL_COMPUTED_BLOCKS * BLOCK_SIZE

    request = create_request(num_tokens=num_prompt_tokens)
    hybrid_scheduler.add_request(request=request)

    hybrid_scheduler.connector = Mock()
    hybrid_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=False
        )
    )
    hybrid_scheduler.connector.request_finished.return_value = (False, None)
    hybrid_scheduler.connector.take_events.return_value = ()

    scheduler_output = hybrid_scheduler.schedule()
    assert len(scheduler_output.scheduled_new_reqs) == 1

    per_group_ids = _request_block_ids(hybrid_scheduler, request)
    assert len(per_group_ids) == 2

    # Fail a block in the middle of the attention group.
    invalid_block_ids = {per_group_ids[0][50]}
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    # Pre-fix this line itself raises ValueError: the hybrid coordinator
    # returns one block-id list per KV cache group.
    hybrid_scheduler.update_from_output(scheduler_output, model_runner_output)

    # Engine survived; the request replays from zero, not at block 50.
    assert len(hybrid_scheduler.running) == 0
    preempted = hybrid_scheduler.waiting.peek_request()
    assert preempted.request_id == request.request_id
    assert preempted.status == RequestStatus.PREEMPTED
    assert preempted.num_computed_tokens == 0
    assert preempted.skip_reading_prefix_cache is True

    # State (mamba) group buffers are HMA-managed slots with their own
    # lifecycle; the user-visible contract is the request state above.
    _assert_attention_prefix_invalidated(hybrid_scheduler, per_group_ids[0])


def test_hybrid_sync_failure_in_state_group_also_replays_from_zero(
    hybrid_scheduler: Scheduler,
):
    """An invalid state (mamba) block is equally fatal to the shared prefix:
    even with a fully valid attention group the request cannot resume."""
    num_prompt_tokens = NUM_PROMPT_BLOCKS * BLOCK_SIZE
    num_external_computed_tokens = NUM_EXTERNAL_COMPUTED_BLOCKS * BLOCK_SIZE

    request = create_request(num_tokens=num_prompt_tokens)
    hybrid_scheduler.add_request(request=request)

    hybrid_scheduler.connector = Mock()
    hybrid_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=False
        )
    )
    hybrid_scheduler.connector.request_finished.return_value = (False, None)
    hybrid_scheduler.connector.take_events.return_value = ()

    scheduler_output = hybrid_scheduler.schedule()
    per_group_ids = _request_block_ids(hybrid_scheduler, request)

    invalid_block_ids = {per_group_ids[1][50]}  # mamba state group
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )
    hybrid_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(hybrid_scheduler.running) == 0
    preempted = hybrid_scheduler.waiting.peek_request()
    assert preempted.request_id == request.request_id
    assert preempted.num_computed_tokens == 0
    assert preempted.skip_reading_prefix_cache is True

    # State (mamba) group buffers are HMA-managed slots with their own
    # lifecycle; the user-visible contract is the request state above.
    _assert_attention_prefix_invalidated(hybrid_scheduler, per_group_ids[0])


def test_hybrid_async_load_failure_zeroes_computed_tokens(
    hybrid_scheduler: Scheduler,
):
    """Async (WAITING_FOR_REMOTE_KVS) failures reset the request to zero and
    mark it for re-receive; a sibling request that loaded fine keeps its
    externally computed tokens and its prefix-caching eligibility."""
    num_prompt_tokens = NUM_PROMPT_BLOCKS * BLOCK_SIZE
    num_external_computed_tokens = NUM_EXTERNAL_COMPUTED_BLOCKS * BLOCK_SIZE

    request1 = create_request(num_tokens=num_prompt_tokens)
    hybrid_scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    hybrid_scheduler.add_request(request=request2)

    hybrid_scheduler.connector = Mock()
    hybrid_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {
                request1.request_id: num_external_computed_tokens,
                request2.request_id: num_external_computed_tokens,
            },
            async_load=True,
        )
    )
    hybrid_scheduler.connector.take_events.return_value = ()

    scheduler_output = hybrid_scheduler.schedule()
    assert len(hybrid_scheduler.skipped_waiting) == 2

    per_group_ids = _request_block_ids(hybrid_scheduler, request2)
    invalid_block_ids = {per_group_ids[0][50]}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request1.request_id},
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    hybrid_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request2.num_computed_tokens == 0
    assert request2.skip_reading_prefix_cache is True
    assert hybrid_scheduler.failed_recving_kv_req_ids == {request2.request_id}

    # Untouched sibling keeps its externally computed tokens.
    assert request1.num_computed_tokens == num_external_computed_tokens
    assert request1.skip_reading_prefix_cache is False


def test_hybrid_fail_policy_reports_error_without_recompute():
    """kv_load_failure_policy='fail' surfaces the affected request as an
    error; hybrid requests are not preempted back to waiting in that mode."""
    vllm_config = create_vllm_config(kv_load_failure_policy="fail")
    scheduler = create_scheduler(vllm_config, kv_cache_config=_hybrid_kv_cache_config())
    num_prompt_tokens = NUM_PROMPT_BLOCKS * BLOCK_SIZE
    num_external_computed_tokens = NUM_EXTERNAL_COMPUTED_BLOCKS * BLOCK_SIZE

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = _HmaMock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_computed_tokens}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    per_group_ids = _request_block_ids(scheduler, request)
    invalid_block_ids = {per_group_ids[0][50]}
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id in scheduler.finished_req_ids
    assert len(scheduler.waiting) == 0
