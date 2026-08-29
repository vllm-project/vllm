# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side accounting tests for KV compression (KeyDiff).

Verifies that the per-request discarded-KV count reported by the model
runner via ModelRunnerOutput flows into Request.num_kv_discarded and
reduces block allocation, without touching logical token accounting.
"""

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def _make_output(
    scheduler_output,
    reqs,
    kv_compression_discarded=None,
):
    req_ids = [r.request_id for r in reqs]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[100] for _ in req_ids],
        kv_compression_discarded=kv_compression_discarded,
    )


def _num_blocks(scheduler, req_id):
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    return len(manager.req_to_blocks[req_id])


def test_kv_compression_discarded_accounting():
    block_size = 16
    scheduler = create_scheduler(block_size=block_size)
    requests = create_requests(num_requests=1, num_tokens=100, max_tokens=200)
    request = requests[0]
    scheduler.add_request(request)

    # Prefill step: all 100 prompt tokens scheduled, 7 blocks allocated.
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens[request.request_id] == 100
    assert request.num_computed_tokens == 100
    assert _num_blocks(scheduler, request.request_id) == 7

    # The model runner reports that full-replacement compaction discarded
    # 50 KV entries at the end of prefill.
    output = _make_output(
        scheduler_output,
        requests,
        kv_compression_discarded={request.request_id: 50},
    )
    scheduler.update_from_output(scheduler_output, output)
    assert request.num_kv_discarded == 50
    # Logical accounting is untouched.
    assert request.num_computed_tokens == 100
    assert request.num_output_tokens == 1

    # Decode for 40 steps. Physical occupancy starts at 50, so cache
    # positions 50..89 reuse the already-allocated blocks: no new block
    # is allocated even though logical length grows to 140 (which would
    # need ceil(141/16) = 9 blocks without compression).
    for _ in range(40):
        scheduler_output = scheduler.schedule()
        assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
        output = _make_output(scheduler_output, requests)
        scheduler.update_from_output(scheduler_output, output)
    assert request.num_computed_tokens == 140
    assert _num_blocks(scheduler, request.request_id) == 7

    # ~70 more decode steps: occupancy crosses 7 * 16 = 112 and a new
    # block is finally needed (at logical 100 + 62 = physical 112).
    for _ in range(70):
        scheduler_output = scheduler.schedule()
        output = _make_output(scheduler_output, requests)
        scheduler.update_from_output(scheduler_output, output)
    assert request.num_computed_tokens == 210
    # The last decode step needed slots for physical occupancy
    # 159 + 1 = 160 -> 10 blocks (without compression, the same step
    # would have needed ceil(210/16) = 14 blocks).
    assert _num_blocks(scheduler, request.request_id) == 10


def test_kv_compression_filtering_accounting():
    block_size = 16
    scheduler = create_scheduler(block_size=block_size)
    requests = create_requests(num_requests=1, num_tokens=32, max_tokens=100)
    request = requests[0]
    scheduler.add_request(request)

    # Prefill: 32 tokens = 2 full blocks.
    scheduler_output = scheduler.schedule()
    output = _make_output(scheduler_output, requests)
    scheduler.update_from_output(scheduler_output, output)
    assert _num_blocks(scheduler, request.request_id) == 2

    # Decode with every other token skipped by online filtering.
    for step in range(32):
        scheduler_output = scheduler.schedule()
        skipped = {request.request_id: 1} if step % 2 == 0 else None
        output = _make_output(
            scheduler_output, requests, kv_compression_discarded=skipped
        )
        scheduler.update_from_output(scheduler_output, output)

    assert request.num_computed_tokens == 64
    assert request.num_kv_discarded == 16
    # The last decode step needed slots for physical occupancy
    # 32 + 31 - 16 + 1 = 48 -> 3 blocks (uncompressed, the same step
    # would have needed ceil(64/16) = 4 blocks).
    assert _num_blocks(scheduler, request.request_id) == 3


def test_kv_compression_reset_on_preemption():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=1, num_tokens=64, max_tokens=10)
    request = requests[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    output = _make_output(
        scheduler_output,
        requests,
        kv_compression_discarded={request.request_id: 32},
    )
    scheduler.update_from_output(scheduler_output, output)
    assert request.num_kv_discarded == 32

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.num_kv_discarded == 0
