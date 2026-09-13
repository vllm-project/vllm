# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV load failures on a hybrid model (several KV cache groups) cannot rewind
the request in place; the request starts over and re-adopts whatever prefix
the prefix cache still offers."""

import pytest
import torch

from tests.v1.kv_connector.unit.utils import MockKVConfig, create_model_runner_output
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

BLOCK_SIZE = 16
WINDOW = 32
NUM_PROMPT_TOKENS = 130
MATCHED = 64
FULL = 0


def _second_group_spec(kind: str) -> KVCacheSpec:
    common = dict(block_size=BLOCK_SIZE, num_kv_heads=1, dtype=torch.float32)
    if kind == "full":
        # A second dense group (different head size) keeps every position.
        return FullAttentionSpec(head_size=2, **common)
    return SlidingWindowSpec(head_size=1, sliding_window=WINDOW, **common)


def _hybrid_scheduler(kv_config: MockKVConfig, second_group: str) -> Scheduler:
    base = create_scheduler(
        block_size=BLOCK_SIZE, enable_prefix_caching=True, use_kv_connector=kv_config
    )
    vllm_config = base.vllm_config
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(["second"], _second_group_spec(second_group)),
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.use_v2_model_runner = True
    scheduler.recompute_kv_load_failures = True
    assert not scheduler.kv_load_failure_rewinds_in_place
    return scheduler


def _new_req_data(out, request):
    return next(r for r in out.scheduled_new_reqs if r.req_id == request.request_id)


def _failed_block(scheduler: Scheduler, request, block_idx: int) -> int:
    blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[FULL]
    return blocks[block_idx].block_id


def _admit_then_fail_sync(scheduler: Scheduler, failed_idx: int):
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert _new_req_data(out, request).num_computed_tokens == MATCHED

    failed = _failed_block(scheduler, request, failed_idx)
    scheduler.update_from_output(
        out, create_model_runner_output([request], invalid_block_ids={failed})
    )
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert not scheduler.running
    return request


@pytest.mark.parametrize("failed_idx", [2, 3])
def test_sync_load_failure_keeps_the_valid_prefix(failed_idx: int):
    """The running request is preempted; its next admission hits exactly the
    blocks before the failed one and loads the rest again."""
    scheduler = _hybrid_scheduler(
        MockKVConfig(matched_tokens=MATCHED, is_async=False), second_group="full"
    )
    request = _admit_then_fail_sync(scheduler, failed_idx)

    out = scheduler.schedule()
    valid = failed_idx * BLOCK_SIZE
    new_req = _new_req_data(out, request)
    assert new_req.num_computed_tokens == valid + MATCHED
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS - (
        valid + MATCHED
    )


def test_sync_load_failure_with_a_window_group_restarts_from_the_load():
    """A sliding-window group holds no blocks below its window, so the hybrid
    lookup offers nothing; the re-admitted request relies on the connector."""
    scheduler = _hybrid_scheduler(
        MockKVConfig(matched_tokens=MATCHED, is_async=False), second_group="swa"
    )
    request = _admit_then_fail_sync(scheduler, failed_idx=2)

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert new_req.num_computed_tokens == MATCHED
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS - MATCHED


def test_async_load_failure_readmits_the_request():
    """A partially failed async load keeps its valid prefix in the cache and
    re-admits the request through the prefix-cache lookup."""
    scheduler = _hybrid_scheduler(
        MockKVConfig(matched_tokens=MATCHED, is_async=True), second_group="full"
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    failed = _failed_block(scheduler, request, 2)
    scheduler.update_from_output(
        out,
        create_model_runner_output(
            [], finished_recving={request.request_id}, invalid_block_ids={failed}
        ),
    )
    assert request.num_computed_tokens == 2 * BLOCK_SIZE

    # Re-admission: the valid prefix is hit locally, the rest is loaded again.
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 2 * BLOCK_SIZE + MATCHED
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )
    out = scheduler.schedule()
    assert _new_req_data(out, request).num_computed_tokens == 2 * BLOCK_SIZE + MATCHED
