# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.v1.core.utils import create_requests, create_scheduler


def test_waiting_kv_blocked_request_does_not_block_lighter_request():
    scheduler = create_scheduler(
        max_num_batched_tokens=8,
        num_blocks=1,
        block_size=4,
        enable_chunked_prefill=True,
    )

    heavy = create_requests(
        num_requests=1, num_tokens=9, block_size=4, req_ids=["heavy"]
    )[0]
    light = create_requests(
        num_requests=1, num_tokens=4, block_size=4, req_ids=["light"]
    )[0]

    scheduler.add_request(heavy)
    scheduler.add_request(light)

    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks

    def _allocate_slots(request, *args, **kwargs):
        if request.request_id == "heavy":
            return None
        return empty_blocks

    scheduler.kv_cache_manager.allocate_slots = _allocate_slots  # type: ignore[method-assign]

    output = scheduler.schedule()
    scheduled_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_ids == ["light"]
